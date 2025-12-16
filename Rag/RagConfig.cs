using System;

namespace LocalLLMChat.Rag;

public class RagConfig
{
    public string DocumentsRootPath { get; set; } = "~/App_Data/RagDocs";
    public string SqliteDbPath { get; set; } = "~/App_Data/rag/rag.db";
    // Default to local GGUF embedding model
    public string EmbeddingModelPath { get; set; } = "~/App_Data/all-MiniLM-L6-v2-Q5_K_M.gguf";
    public int ChunkMaxChars { get; set; } = 1600;
    public int ChunkOverlapChars { get; set; } = 250;
    public int TopK { get; set; } = 5;
    public int MaxContextChars { get; set; } = 6000;
    public bool EnableCitations { get; set; } = true;
}
