using System.Text.Json;
using LocalLLMChat.Models;
using Microsoft.Extensions.Options;

namespace LocalLLMChat.Services;

public class JsonLlmSettingsStore : ILlmSettingsStore
{
    private readonly string _filePath;
    private readonly ILogger<JsonLlmSettingsStore> _logger;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private readonly LlmSettings _defaultSettings;

    public JsonLlmSettingsStore(IHostEnvironment env, IOptions<LlmSettings> defaults, ILogger<JsonLlmSettingsStore> logger)
    {
        _logger = logger;
        _defaultSettings = defaults.Value ?? new LlmSettings();

        var appData = Path.Combine(env.ContentRootPath, "App_Data");
        Directory.CreateDirectory(appData);
        _filePath = Path.Combine(appData, "llmsettings.json");
    }

    public LlmSettings Get()
    {
        _lock.Wait();
        try
        {
            if (File.Exists(_filePath))
            {
                var json = File.ReadAllText(_filePath);
                var settings = JsonSerializer.Deserialize<LlmSettings>(json);
                if (settings != null)
                {
                    return Normalize(settings);
                }
            }

            return Normalize(_defaultSettings.Clone());
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Failed to read LLM settings file, falling back to defaults.");
            return Normalize(_defaultSettings.Clone());
        }
        finally
        {
            _lock.Release();
        }
    }

    public void Save(LlmSettings settings)
    {
        _lock.Wait();
        try
        {
            var normalized = Normalize(settings);
            var json = JsonSerializer.Serialize(normalized, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(_filePath, json);
        }
        finally
        {
            _lock.Release();
        }
    }

    private static LlmSettings Normalize(LlmSettings settings)
    {
        // Enforce dependent defaults
        if (settings.CpuOnly)
        {
            settings.GpuOffloadLayers = 0;
            settings.OffloadKvCacheToGpu = false;
        }

        return settings;
    }
}
