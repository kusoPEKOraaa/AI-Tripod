/**
 * AI-Tripod API 客户端
 * 封装所有后端 API 调用
 */

const API = {
    // API 基础路径
    baseUrl: '/api',
    
    // 获取认证 token
    getToken() {
        return localStorage.getItem('token');
    },
    
    // 设置认证 token
    setToken(token) {
        localStorage.setItem('token', token);
    },
    
    // 清除认证 token
    clearToken() {
        localStorage.removeItem('token');
    },
    
    // 检查是否已登录
    isLoggedIn() {
        return !!this.getToken();
    },
    
    // 通用请求方法
    async request(endpoint, options = {}) {
        const url = endpoint.startsWith('http') ? endpoint : `${this.baseUrl}${endpoint}`;
        
        const headers = {
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        // 添加认证 token
        const token = this.getToken();
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        
        const config = {
            ...options,
            headers
        };
        
        try {
            const response = await fetch(url, config);
            
            // 401 未授权处理
            if (response.status === 401) {
                this.clearToken();
                window.location.hash = '#/login';
                throw new Error('登录已过期，请重新登录');
            }
            
            // 处理响应
            const contentType = response.headers.get('content-type');
            let data;
            
            if (contentType && contentType.includes('application/json')) {
                data = await response.json();
            } else {
                data = await response.text();
            }
            
            if (!response.ok) {
                // 处理错误信息，可能是字符串或对象数组
                let errorMessage = '请求失败';
                if (data.detail) {
                    if (typeof data.detail === 'string') {
                        errorMessage = data.detail;
                    } else if (Array.isArray(data.detail)) {
                        // FastAPI 验证错误格式
                        errorMessage = data.detail.map(e => e.msg || e.message || JSON.stringify(e)).join('; ');
                    } else if (typeof data.detail === 'object') {
                        errorMessage = data.detail.msg || data.detail.message || JSON.stringify(data.detail);
                    }
                } else if (data.message) {
                    errorMessage = data.message;
                }
                const error = new Error(errorMessage);
                error.status = response.status;
                error.data = data;
                throw error;
            }
            
            return data;
        } catch (error) {
            if (error.name === 'TypeError') {
                throw new Error('网络错误，请检查网络连接');
            }
            throw error;
        }
    },
    
    // GET 请求
    get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    },
    
    // POST 请求
    post(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    // PUT 请求
    put(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    // DELETE 请求
    delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    },
    
    // 表单提交（用于登录）
    async postForm(endpoint, data) {
        const url = endpoint.startsWith('http') ? endpoint : endpoint;
        const formData = new URLSearchParams(data);
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: formData
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || '请求失败');
        }
        
        return result;
    },

    // =========== 认证相关 ===========
    auth: {
        // 登录
        async login(username, password, captcha, captchaId) {
            const data = await API.postForm('/token', {
                username,
                password,
                captcha,
                captcha_id: captchaId
            });
            API.setToken(data.access_token);
            return data;
        },
        
        // 注册
        async register(userData) {
            return API.post('/register', userData);
        },
        
        // 获取验证码
        getCaptchaUrl() {
            return `/api/captcha?t=${Date.now()}`;
        },
        
        // 登出
        logout() {
            API.clearToken();
            window.location.hash = '#/login';
        },
        
        // 获取当前用户信息
        getCurrentUser() {
            return API.get('/users/me');
        }
    },
    
    // =========== 资源管理 ===========
    resources: {
        // 获取资源列表
        list(params = {}) {
            return API.get('/resources', params);
        },
        
        // 获取资源详情
        get(id) {
            return API.get(`/resources/${id}`);
        },
        
        // 创建资源
        create(data) {
            return API.post('/resources', data);
        },
        
        // 更新资源
        update(id, data) {
            return API.put(`/resources/${id}`, data);
        },
        
        // 删除资源
        delete(id) {
            return API.delete(`/resources/${id}`);
        },
        
        // 开始下载
        startDownload(id, source = 'OFFICIAL') {
            return API.post(`/resources/${id}/download`, { source });
        },
        
        // 停止下载
        stopDownload(id) {
            return API.post(`/resources/${id}/stop`);
        },
        
        // 重试下载
        retryDownload(id) {
            return API.post(`/resources/${id}/retry`);
        },
        
        // 刷新状态
        refresh(id) {
            return API.post(`/resources/${id}/refresh`);
        }
    },
    
    // =========== 训练任务 ===========
    training: {
        // 获取任务列表
        list(params = {}) {
            return API.get('/training/tasks', params);
        },
        
        // 获取任务详情
        get(id) {
            return API.get(`/training/tasks/${id}`);
        },
        
        // 创建任务
        create(data) {
            return API.post('/training/tasks', data);
        },
        
        // 删除任务
        delete(id) {
            return API.delete(`/training/tasks/${id}`);
        },
        
        // 启动任务
        start(id) {
            return API.post(`/training/tasks/${id}/start`);
        },
        
        // 停止任务
        stop(id) {
            return API.post(`/training/tasks/${id}/stop`);
        },
        
        // 获取日志
        getLogs(id) {
            return API.get(`/training/tasks/${id}/logs`);
        },
        
        // WebSocket 连接日志
        connectLogs(id, onMessage, onError) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${window.location.host}/ws/training/${id}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                onMessage(data);
            };
            
            ws.onerror = (error) => {
                if (onError) onError(error);
            };
            
            return ws;
        }
    },
    
    // =========== 推理服务 ===========
    inference: {
        // 获取任务列表
        list(params = {}) {
            return API.get('/inference/tasks', params);
        },
        
        // 获取任务详情
        get(id) {
            return API.get(`/inference/tasks/${id}`);
        },
        
        // 创建任务
        create(data) {
            return API.post('/inference/tasks', data);
        },
        
        // 删除任务
        delete(id) {
            return API.delete(`/inference/tasks/${id}`);
        },
        
        // 启动服务
        start(id) {
            return API.post(`/inference/tasks/${id}/start`);
        },
        
        // 停止服务
        stop(id) {
            return API.post(`/inference/tasks/${id}/stop`);
        },
        
        // 发送聊天消息
        chat(id, message) {
            return API.post(`/inference/tasks/${id}/chat`, message);
        },
        
        // 获取 GPU 状态
        getGpuStatus() {
            return API.get('/inference/gpu');
        },
        
        // WebSocket 聊天连接
        connectChat(id, onMessage, onError) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const token = API.getToken();
            const ws = new WebSocket(`${protocol}//${window.location.host}/api/ws/chat/${id}?token=${token}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                onMessage(data);
            };
            
            ws.onerror = (error) => {
                if (onError) onError(error);
            };
            
            return ws;
        }
    },
    
    // =========== 评估任务 ===========
    evaluation: {
        // 获取任务列表
        list(params = {}) {
            return API.get('/evaluation/tasks', params);
        },
        
        // 获取任务详情
        get(id) {
            return API.get(`/evaluation/tasks/${id}`);
        },
        
        // 创建任务
        create(data) {
            return API.post('/evaluation/tasks', data);
        },
        
        // 删除任务
        delete(id) {
            return API.delete(`/evaluation/tasks/${id}`);
        },
        
        // 启动任务
        start(id) {
            return API.post(`/evaluation/tasks/${id}/start`);
        },
        
        // 停止任务
        stop(id) {
            return API.post(`/evaluation/tasks/${id}/stop`);
        },
        
        // 获取日志
        getLogs(id) {
            return API.get(`/evaluation/tasks/${id}/logs`);
        },
        
        // WebSocket 连接日志
        connectLogs(id, onMessage, onError) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const ws = new WebSocket(`${protocol}//${window.location.host}/ws/evaluation/${id}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                onMessage(data);
            };
            
            ws.onerror = (error) => {
                if (onError) onError(error);
            };
            
            return ws;
        }
    },
    
    // =========== 用户管理（管理员） ===========
    users: {
        // 获取用户列表
        list(params = {}) {
            return API.get('/users', params);
        },
        
        // 创建用户
        create(data) {
            return API.post('/users', data);
        },
        
        // 删除用户
        delete(id) {
            return API.delete(`/users/${id}`);
        }
    },
    
    // =========== 个人资料 ===========
    profile: {
        // 获取个人资料
        get() {
            return API.get('/profile');
        },
        
        // 更新个人资料
        update(data) {
            return API.put('/profile', data);
        },
        
        // 修改密码
        changePassword(data) {
            return API.post('/profile/change-password', data);
        },
        
        // 上传头像
        async uploadAvatar(file) {
            const formData = new FormData();
            formData.append('file', file);
            
            const token = API.getToken();
            const response = await fetch('/api/profile/avatar', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${token}`
                },
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || '上传失败');
            }
            
            return result;
        }
    },
    
    // =========== 分享 ===========
    share: {
        // 获取分享列表
        list(params = {}) {
            return API.get('/share', params);
        },
        
        // 获取分享详情
        get(id) {
            return API.get(`/share/${id}`);
        }
    }
};

// 导出
window.API = API;
