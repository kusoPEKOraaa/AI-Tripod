import { useRouteError, useNavigate } from 'react-router-dom'
import { Button, Result } from 'antd'

export const RouteErrorPage = () => {
  const error = useRouteError() as Error
  const navigate = useNavigate()

  return (
    <div style={{ padding: '50px', textAlign: 'center' }}>
      <Result
        status="error"
        title="页面加载失败"
        subTitle={error?.message || '未知错误'}
        extra={[
          <Button type="primary" key="back" onClick={() => navigate(-1)}>
            返回上一页
          </Button>,
          <Button key="home" onClick={() => navigate('/')}>
            返回首页
          </Button>,
        ]}
      />
    </div>
  )
}
