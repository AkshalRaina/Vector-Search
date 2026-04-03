import "dotenv/config";
import fs from "fs";
import csv from "csv-parser";
import { pipeline } from "@xenova/transformers";
import { Client } from "@opensearch-project/opensearch";

const INDEX_NAME = "dataset";
const BATCH_SIZE = 50;

const client = new Client({
  node: process.env.OPENSEARCH_URL,
  auth: {
    username: process.env.OPENUSERNAME,
    password: process.env.PASSWORD,
  },
  ssl: { rejectUnauthorized: false },
});

async function ingestData() {
  console.log("--- Initializing AI Model ---");

  const embedder = await pipeline(
    "feature-extraction",
    "Xenova/all-mpnet-base-v2"
  );

  let operations = [];
  let count = 0;

  const stream = fs.createReadStream("myntra_products_catalog.csv").pipe(csv());

  for await (const row of stream) {
    try {
      if (!row.Description) continue;

      const output = await embedder(row.Description, {
        pooling: "mean",
        normalize: true,
      });

      const vector = Array.from(output.data);

      if (vector.length !== 768) {
        console.error(
          `ID ${row.ProductID}: Wrong dimension (${vector.length})`
        );
        continue;
      }

      operations.push({ index: { _index: INDEX_NAME } });
      operations.push({
        ProductID: parseInt(row.ProductID),
        ProductName: row.ProductName,
        ProductBrand: row.ProductBrand,
        Gender: row.Gender,
        Price: parseInt(row.Price),
        NumImages: parseInt(row.NumImages),
        PrimaryColor: row.PrimaryColor,
        Description: row.Description,
        DescriptionVector: vector,
      });

      count++;

      if (operations.length >= BATCH_SIZE * 2) {
        const bulkResponse = await client.bulk({ body: operations });

        if (bulkResponse.body?.errors) {
          const erroredItem = bulkResponse.body.items.find(
            (item) => item.index && item.index.error
          );

          console.error(
            "Bulk Error:",
            JSON.stringify(erroredItem.index.error, null, 2)
          );
        } else {
          console.log(`Successfully indexed ${count} documents...`);
        }

        operations = [];
      }
    } catch (err) {
      console.error(
        `Skipping Product ${row.ProductID} due to error:`,
        err.message
      );
    }
  }

  // Final flush
  if (operations.length > 0) {
    const bulkResponse = await client.bulk({ body: operations });

    if (bulkResponse.body?.errors) {
      console.error("Final batch had errors");
    } else {
      console.log("Final batch processed.");
    }
  }

  console.log(`--- Ingestion Complete! Total: ${count} ---`);
}

ingestData().catch(console.error);