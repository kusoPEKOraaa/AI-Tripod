import { Button, Flex, Space, Tag, Tooltip, Typography, Select } from 'antd'
import { QuestionCircleOutlined } from '@ant-design/icons'
import { useLocation, useNavigate } from 'react-router-dom'
import { useEffect } from 'react'
import { workspaceNavItems } from '@/constants/navigation'
import { supportedLanguages } from '@/lib/i18n'
import { useGlobalConfigStore } from '@/stores/useGlobalConfig'
import i18n from '@/lib/i18n'

export const AppTopbar = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const language = useGlobalConfigStore((state) => state.language)
  const setLanguage = useGlobalConfigStore((state) => state.setLanguage)
  const currentNav = workspaceNavItems.find((item) => location.pathname.startsWith(item.path))

  // 同步 zustand store 和 i18n 语言设置
  useEffect(() => {
    if (i18n.language !== language) {
      i18n.changeLanguage(language)
    }
  }, [language])

  return (
    <header className="workspace-topbar">
      <Flex justify="space-between" align="center">
        <div>
          <Typography.Title level={4} style={{ margin: 0 }}>
            {currentNav?.label ?? 'Workspace'}
          </Typography.Title>
          <Typography.Text type="secondary">{currentNav?.description}</Typography.Text>
        </div>
        <Space size="middle">
          <Tag color="purple">Prototype</Tag>
          <Select
            size="small"
            value={language}
            options={supportedLanguages}
            onChange={(value) => setLanguage(value as 'zh' | 'en')}
          />
          <Tooltip title="查看最新使用文档">
            <Button
              icon={<QuestionCircleOutlined />}
              href="https://github.com/hiyouga/LLaMA-Factory"
              target="_blank"
            >
              Docs
            </Button>
          </Tooltip>
          <Button type="primary" onClick={() => navigate('/app/quick-start')}>
            New Task
          </Button>
        </Space>
      </Flex>
    </header>
  )
}
