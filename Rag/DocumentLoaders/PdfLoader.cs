using System.IO;
using System.Text;
using PdfDocument = UglyToad.PdfPig.PdfDocument;

namespace LocalLLMChat.Rag.DocumentLoaders;

public class PdfLoader : ITextDocumentLoader
{
    public IEnumerable<(string sourceId, string sourcePath, string text)> LoadAll(string rootPath)
    {
        if (!Directory.Exists(rootPath))
        {
            yield break;
        }

        foreach (var file in Directory.EnumerateFiles(rootPath, "*.pdf", SearchOption.AllDirectories))
        {
            var text = ExtractSafely(file);
            if (text == null) continue;

            var sourceId = Path.GetFileName(file);
            yield return (sourceId, file, text);
        }
    }

    private static string ExtractText(string path)
    {
        var sb = new StringBuilder();
        using var doc = PdfDocument.Open(path);
        foreach (var page in doc.GetPages())
        {
            sb.AppendLine($"[Page {page.Number}]");
            sb.AppendLine(page.Text);
            sb.AppendLine();
        }
        return sb.ToString();
    }

    private static string? ExtractSafely(string path)
    {
        try
        {
            return ExtractText(path);
        }
        catch
        {
            return null;
        }
    }
}
