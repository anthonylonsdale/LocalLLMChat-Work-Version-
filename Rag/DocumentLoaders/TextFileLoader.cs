using System.IO;

namespace LocalLLMChat.Rag.DocumentLoaders;

public class TextFileLoader : ITextDocumentLoader
{
    private static readonly string[] Extensions = new[] { ".txt", ".md" };

    public IEnumerable<(string sourceId, string sourcePath, string text)> LoadAll(string rootPath)
    {
        if (!Directory.Exists(rootPath))
        {
            yield break;
        }

        foreach (var file in Directory.EnumerateFiles(rootPath, "*.*", SearchOption.AllDirectories)
                                      .Where(f => Extensions.Contains(Path.GetExtension(f), StringComparer.OrdinalIgnoreCase)))
        {
            string text;
            try
            {
                text = File.ReadAllText(file);
            }
            catch
            {
                continue;
            }

            var sourceId = Path.GetFileName(file);
            yield return (sourceId, file, text);
        }
    }
}
