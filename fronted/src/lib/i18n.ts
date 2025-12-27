import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

export const supportedLanguages = [
  { label: '中文', value: 'zh' },
  { label: 'English', value: 'en' },
]

i18n.use(initReactI18next).init({
  resources: {
    en: {
      translation: {
        landing: {
          title: 'Unified Fine-tuning Workspace',
          subtitle: 'Launch and monitor LLM/Multi-modal jobs from one coherent console.',
          cta: 'Open Console',
        },
      },
    },
    zh: {
      translation: {
        landing: {
          title: '大模型集成训练工作台',
          subtitle: '在统一界面发起与监控多模态训练与推理任务。',
          cta: '进入控制台',
        },
      },
    },
  },
  lng: supportedLanguages[0].value,
  fallbackLng: 'en',
  interpolation: {
    escapeValue: false,
  },
})

export default i18n
