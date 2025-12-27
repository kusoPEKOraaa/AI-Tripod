import { useMemo } from 'react'
import { ConfigProvider, App as AntdApp, theme as antdTheme } from 'antd'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { StyleProvider } from '@ant-design/cssinjs'
import type { ReactNode } from 'react'
import { ErrorBoundary } from '@/components/error/ErrorBoundary'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      staleTime: 1000 * 30,
    },
  },
})

type AppProvidersProps = {
  children: ReactNode
}

export const AppProviders = ({ children }: AppProvidersProps) => {
  const theme = useMemo(
    () => ({
      token: {
        colorPrimary: '#5b34da',
        colorBgLayout: '#f8fafc',
        colorText: '#0f172a',
        borderRadius: 10,
      },
      algorithm: antdTheme.defaultAlgorithm,
    }),
    [],
  )

  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <StyleProvider hashPriority="high">
          <ConfigProvider theme={theme}>
            <AntdApp>{children}</AntdApp>
          </ConfigProvider>
        </StyleProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  )
}
