    import 'dotenv/config';
    import express from 'express';
    import path from 'path';
    import { fileURLToPath } from 'url';
    import { Client } from '@opensearch-project/opensearch';
    import { pipeline } from '@xenova/transformers';

    const app = express();
    const PORT = 3000;
    const INDEX_NAME = 'dataset';

    // Middleware to parse JSON and URL-encoded bodies
    app.use(express.json());
    app.use(express.urlencoded({ extended: true }));
    app.use(express.static('public'));
    // Fix for __dirname in ES Modules
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);

    // 1. Initialize OpenSearch Client
    const client = new Client({
        node: process.env.OPENSEARCH_URL, 
        auth: { 
            username: process.env.OPENUSERNAME, 
            password: process.env.PASSWORD 
        },
        ssl: { rejectUnauthorized: false }
    });

    // 2. Load the Embedding Model
    let embedder;
    async function loadModel() {
        console.log("--- Loading AI Model ---");
        embedder = await pipeline('feature-extraction', 'Xenova/all-mpnet-base-v2');
        console.log("--- Model Ready ---");
    }

    // Serve HTML form at root route
    app.get('/', (req, res) => {
        res.sendFile(path.join(__dirname, 'index.html'));
    });

    // 3. Search Route (POST)
    app.post('/api/search', async (req, res) => {
        try {
            const { query } = req.body;
            if (!query) return res.status(400).json({ error: "Query is required" });

            if (!embedder) await loadModel();

            // 1. Convert search text to vector
            const output = await embedder(query, { pooling: 'mean', normalize: true });
            const queryVector = Array.from(output.data);

            // 2. Pure K-NN Search Query
            // This ignores literal word matches and focuses entirely on semantic distance
            const body = {
                size: 15,
                query: {
                    knn: {
                        "DescriptionVector": {
                            "vector": queryVector,
                            "k": 15 // Returns the 15 closest mathematical neighbors
                        }
                    }
                }
            };

            const result = await client.search({ index: INDEX_NAME, body });
            
            const products = result.body.hits.hits.map(hit => ({
                id: hit._id,
                productId: hit._source.ProductID,
                score: hit._score, // In K-NN, higher score = mathematically closer
                title: hit._source.ProductName,
                description: hit._source.Description,
                price: hit._source.Price
            }));

            res.json({ results: products });
        } catch (error) {
            console.error("Search error:", error.message);
            res.status(500).json({ error: "Search failed: " + error.message });
        }
    });

    // 4. Start Server
    loadModel().then(() => {
        app.listen(PORT, () => {
            console.log(`Server running at http://localhost:${PORT}`);
        });
    });
