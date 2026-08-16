import os
import unittest

from llama_index.embeddings.openai import OpenAIEmbedding


class MainTest(unittest.TestCase):
    def test_initializes_embedding_model_from_env(self):
        import main

        self.assertIsInstance(main.embedding_model, OpenAIEmbedding)
        self.assertEqual(main.embedding_model.model_name, os.environ["LLM_EMBEDDING_MODEL"])
        self.assertEqual(main.embedding_model.api_base, os.environ["LLM_BASE_URL"])


if __name__ == "__main__":
    unittest.main()
