using LocalLLMChat.Rag;
using Microsoft.AspNetCore.Mvc;

namespace LocalLLMChat.Controllers;

[Route("Rag")]
public class RagController : Controller
{
    private readonly RagPipeline _pipeline;
    private readonly ILogger<RagController> _logger;

    public RagController(RagPipeline pipeline, ILogger<RagController> logger)
    {
        _pipeline = pipeline;
        _logger = logger;
    }

    [HttpPost("Ingest")]
    public async Task<IActionResult> Ingest(CancellationToken cancellationToken)
    {
        try
        {
            await _pipeline.IngestAsync(cancellationToken);
            return Ok(new { ok = true });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "RAG ingestion failed");
            return StatusCode(500, new { ok = false, error = ex.Message });
        }
    }
}
