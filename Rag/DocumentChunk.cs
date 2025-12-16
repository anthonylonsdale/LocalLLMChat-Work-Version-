using System;

namespace LocalLLMChat.Rag;

public class DocumentChunk
{
    public string ChunkId { get; set; } = string.Empty;
    public string SourceId { get; set; } = string.Empty;
    public string SourcePath { get; set; } = string.Empty;
    public int ChunkIndex { get; set; }
    public string Text { get; set; } = string.Empty;
    public DateTimeOffset IngestedAt { get; set; } = DateTimeOffset.UtcNow;
}
