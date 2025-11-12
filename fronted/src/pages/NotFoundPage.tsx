import { Button, Result } from 'antd'
import { useNavigate } from 'react-router-dom'

export const NotFoundPage = () => {
  const navigate = useNavigate()
  return (
    <Result
      status="404"
      title="页面不存在"
      subTitle="请确认链接是否正确或返回控制台。"
      extra={
        <Button type="primary" onClick={() => navigate('/app/quick-start')}>
          返回平台
        </Button>
      }
    />
  )
}
