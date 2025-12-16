namespace LocalLLMChat.Services;

/// <summary>
/// Interface for Retrieval-Augmented Generation (RAG) service
/// </summary>
public interface IRagService
{
    /// <summary>
    /// Search for relevant documents based on a query
    /// </summary>
    Task<IEnumerable<RagDocument>> SearchAsync(string query, int topK = 5, CancellationToken cancellationToken = default);

    /// <summary>
    /// Add a document to the knowledge base
    /// </summary>
    Task AddDocumentAsync(string content, Dictionary<string, string>? metadata = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Add documents from a file
    /// </summary>
    Task AddDocumentFromFileAsync(string filePath, CancellationToken cancellationToken = default);

    /// <summary>
    /// Clear all documents from the knowledge base
    /// </summary>
    Task ClearAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Get the count of documents in the knowledge base
    /// </summary>
    Task<int> GetDocumentCountAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Represents a document retrieved from the RAG system
/// </summary>
public class RagDocument
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string Content { get; set; } = string.Empty;
    public float RelevanceScore { get; set; }
    public Dictionary<string, string> Metadata { get; set; } = new();
}
