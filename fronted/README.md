# AI Tripod Frontend

React + TypeScript workspace that recreates the LLaMA‑Factory WebUI experience with Django APIs. The current delivery focuses on the foundation: routing, layout shell, state providers, and placeholder experiences for every major feature (Quick Start, Tasks, Datasets, Chat, Eval, Export, Admin).

## Tech Stack

- [Vite](https://vitejs.dev/) + React 19 + TypeScript
- [Ant Design 5](https://ant.design/) for layout, forms, and theming
- [React Router v7](https://reactrouter.com/) for nested routes
- [Zustand](https://github.com/pmndrs/zustand) for lightweight global settings
- [@tanstack/react-query](https://tanstack.com/query) for async data (wired to local mock APIs for now)
- [i18next](https://www.i18next.com/) ready for multi-language copy (#zh / #en)

## Getting Started

```bash
cd fronted
npm install        # already run once during scaffolding
npm run dev        # start Vite dev server on http://localhost:5173
npm run build      # type-check + production build
npm run preview    # preview production build locally
```

If you need to tweak lint rules, edit `eslint.config.js`.

## Project Layout

```
fronted/
├─ src/
│  ├─ app/                 # router + shared layouts/providers
│  ├─ components/          # reusable UI pieces
│  ├─ pages/               # route-level experiences (Landing, QuickStart, Tasks, etc.)
│  ├─ services/            # API client + mock data sources
│  ├─ stores/              # Zustand stores for global config
│  ├─ constants/           # nav config, enums
│  ├─ lib/                 # i18n bootstrap
│  └─ index.css            # global styling + workspace layout helpers
├─ package.json
├─ tsconfig*.json          # TS + path alias config (`@/*` → `src/*`)
└─ vite.config.ts          # alias + plugin setup
```

## Next Steps

1. **Hook up real APIs**: replace `services/mockApi.ts` with Django endpoints for models, datasets, and tasks (React Query already wraps the calls).
2. **Streaming logs & chat**: connect WebSocket/SSE endpoints to the Task board and Chat Lab placeholders.
3. **Auth integration**: gate `/app/*` routes based on session/JWT and prefetch user-specific quotas.
4. **Form schemas**: generate Quick Start & Advanced editor fields from backend metadata to avoid divergence from LLaMA‑Factory CLI arguments.
5. **Styling polish**: align with your design system tokens (colors/spacing currently follow a dark proto theme).
