import { createBrowserRouter } from 'react-router-dom'
import { LandingPage } from '@/pages/LandingPage'
import { WorkspaceLayout } from '@/app/layout/WorkspaceLayout'
import { QuickStartPage } from '@/pages/QuickStartPage'
import { TasksPage } from '@/pages/TasksPage'
import { DatasetsPage } from '@/pages/DatasetsPage'
import { ChatLabPage } from '@/pages/ChatLabPage'
import { EvalLabPage } from '@/pages/EvalLabPage'
import { ExportHubPage } from '@/pages/ExportHubPage'
import { AdminPage } from '@/pages/AdminPage'
import { NotFoundPage } from '@/pages/NotFoundPage'
import { RouteErrorPage } from '@/pages/RouteErrorPage'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <LandingPage />,
    errorElement: <RouteErrorPage />,
  },
  {
    path: '/app',
    element: <WorkspaceLayout />,
    errorElement: <RouteErrorPage />,
    children: [
      {
        index: true,
        element: <QuickStartPage />,
      },
      {
        path: 'quick-start',
        element: <QuickStartPage />,
      },
      {
        path: 'tasks',
        element: <TasksPage />,
      },
      {
        path: 'datasets',
        element: <DatasetsPage />,
      },
      {
        path: 'chat-lab',
        element: <ChatLabPage />,
      },
      {
        path: 'eval-lab',
        element: <EvalLabPage />,
      },
      {
        path: 'export-hub',
        element: <ExportHubPage />,
      },
      {
        path: 'admin',
        element: <AdminPage />,
      },
    ],
  },
  {
    path: '*',
    element: <NotFoundPage />,
  },
])
