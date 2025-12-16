using LocalLLMChat.Models;

namespace LocalLLMChat.Services;

public interface ILlmSettingsStore
{
    LlmSettings Get();
    void Save(LlmSettings settings);
}
