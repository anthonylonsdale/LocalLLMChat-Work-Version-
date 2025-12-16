namespace LocalLLMChat.Rag.DocumentLoaders;

public interface ITextDocumentLoader
{
    IEnumerable<(string sourceId, string sourcePath, string text)> LoadAll(string rootPath);
}
