import { create } from 'zustand'

type FinetuningType = 'lora' | 'full' | 'qlora'

type GlobalConfigState = {
  selectedModel?: string
  hub: 'huggingface' | 'modelscope' | 'openmind'
  finetuningType: FinetuningType
  language: 'zh' | 'en'
  updateConfig: (payload: Partial<Omit<GlobalConfigState, 'updateConfig' | 'setLanguage'>>) => void
  setLanguage: (language: 'zh' | 'en') => void
}

export const useGlobalConfigStore = create<GlobalConfigState>((set) => ({
  hub: 'huggingface',
  finetuningType: 'lora',
  language: 'zh',
  updateConfig: (payload) => set(payload),
  setLanguage: (language) => set({ language }),
}))
