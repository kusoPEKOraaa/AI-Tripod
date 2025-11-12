type RequestOptions = {
  method?: 'GET' | 'POST'
  body?: Record<string, unknown>
}

const BASE_URL = '/api'

export const httpClient = async <T>(endpoint: string, options: RequestOptions = {}) => {
  const response = await fetch(`${BASE_URL}${endpoint}`, {
    method: options.method ?? 'GET',
    headers: {
      'Content-Type': 'application/json',
    },
    body: options.body ? JSON.stringify(options.body) : undefined,
  })

  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`)
  }

  return (await response.json()) as T
}
