/**
 * AI-Tripod 页面模块
 * 所有页面的渲染和交互逻辑
 */

const Pages = {
    // =========== 首页 ===========
    home: {
        async render() {
            document.body.classList.remove('hide-header');

            const content = `
                <div class="home-page">
                    <!-- 英雄区域 -->
                    <div class="home-hero">
                        <h1 class="home-hero-title">欢迎使用 AI-Tripod</h1>
                        <p class="home-hero-subtitle">
                            大模型集成训练平台，支持模型训练、推理、评估与导出的一体化工作流
                        </p>
                        <div class="home-hero-actions">
                            <a href="#/resources" class="btn btn-primary btn-lg">
                                <span class="material-icons">rocket_launch</span>
                                开始使用
                            </a>
                            <a href="#/dashboard" class="btn btn-outline btn-lg">
                                <span class="material-icons">dashboard</span>
                                查看仪表盘
                            </a>
                        </div>
                    </div>

                    <!-- 统计数据 -->
                    <div class="home-stats" id="home-stats">
                        ${Components.loading('加载统计数据...')}
                    </div>

                    <!-- 快捷操作 -->
                    <div class="home-section">
                        <h2 class="home-section-title">
                            <span class="material-icons">flash_on</span>
                            快捷操作
                        </h2>
                        <div class="home-quick-actions">
                            <a href="#/resources" class="home-quick-action">
                                <span class="material-icons">cloud_download</span>
                                <span class="home-quick-action-title">下载资源</span>
                            </a>
                            <a href="#/training" class="home-quick-action">
                                <span class="material-icons">model_training</span>
                                <span class="home-quick-action-title">创建训练</span>
                            </a>
                            <a href="#/inference" class="home-quick-action">
                                <span class="material-icons">psychology</span>
                                <span class="home-quick-action-title">启动推理</span>
                            </a>
                            <a href="#/evaluation" class="home-quick-action">
                                <span class="material-icons">analytics</span>
                                <span class="home-quick-action-title">模型评估</span>
                            </a>
                        </div>
                    </div>

                    <!-- 功能特性 -->
                    <div class="home-section">
                        <h2 class="home-section-title">
                            <span class="material-icons">stars</span>
                            核心功能
                        </h2>
                        <div class="home-features">
                            <div class="home-feature-card">
                                <div class="home-feature-icon">
                                    <span class="material-icons">model_training</span>
                                </div>
                                <h3 class="home-feature-title">模型训练</h3>
                                <p class="home-feature-description">
                                    支持多种基座模型，灵活配置训练参数，实时监控训练进度与日志
                                </p>
                            </div>
                            <div class="home-feature-card">
                                <div class="home-feature-icon">
                                    <span class="material-icons">psychology</span>
                                </div>
                                <h3 class="home-feature-title">智能推理</h3>
                                <p class="home-feature-description">
                                    一键启动推理服务，支持多模态对话，实时响应用户请求
                                </p>
                            </div>
                            <div class="home-feature-card">
                                <div class="home-feature-icon">
                                    <span class="material-icons">analytics</span>
                                </div>
                                <h3 class="home-feature-title">模型评估</h3>
                                <p class="home-feature-description">
                                    集成多种评估基准，自动生成评估报告，直观展示模型性能
                                </p>
                            </div>
                            <div class="home-feature-card">
                                <div class="home-feature-icon">
                                    <span class="material-icons">folder</span>
                                </div>
                                <h3 class="home-feature-title">资源管理</h3>
                                <p class="home-feature-description">
                                    统一管理模型与数据集，支持从 HuggingFace 一键下载资源
                                </p>
                            </div>
                        </div>
                    </div>

                    <!-- 支持的模型 -->
                    <div class="home-section">
                        <h2 class="home-section-title">
                            <span class="material-icons">verified</span>
                            支持的基础模型
                        </h2>
                        <div class="home-models">
                            <span class="home-model-tag">Qwen2.5</span>
                            <span class="home-model-tag">Llama 3.1</span>
                            <span class="home-model-tag">DeepSeek</span>
                            <span class="home-model-tag">InternLM</span>
                            <span class="home-model-tag">Yi</span>
                            <span class="home-model-tag">Gemma</span>
                            <span class="home-model-tag">Mistral</span>
                            <span class="home-model-tag">Phi-3</span>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('main-content').innerHTML = content;
            this.loadStats();
        },

        async loadStats() {
            try {
                const [resources, training, inference, evaluation] = await Promise.all([
                    API.resources.list(),
                    API.training.list(),
                    API.inference.list(),
                    API.evaluation.list()
                ]);

                const statsHtml = `
                    <div class="home-stat-card">
                        <div class="home-stat-icon">
                            <span class="material-icons">folder</span>
                        </div>
                        <div class="home-stat-value">${resources.length}</div>
                        <div class="home-stat-label">资源总数</div>
                    </div>
                    <div class="home-stat-card">
                        <div class="home-stat-icon">
                            <span class="material-icons">model_training</span>
                        </div>
                        <div class="home-stat-value">${training.length}</div>
                        <div class="home-stat-label">训练任务</div>
                    </div>
                    <div class="home-stat-card">
                        <div class="home-stat-icon">
                            <span class="material-icons">psychology</span>
                        </div>
                        <div class="home-stat-value">${inference.filter(t => t.status === 'RUNNING').length}</div>
                        <div class="home-stat-label">运行中服务</div>
                    </div>
                    <div class="home-stat-card">
                        <div class="home-stat-icon">
                            <span class="material-icons">analytics</span>
                        </div>
                        <div class="home-stat-value">${evaluation.length}</div>
                        <div class="home-stat-label">评估任务</div>
                    </div>
                `;

                document.getElementById('home-stats').innerHTML = statsHtml;
            } catch (error) {
                document.getElementById('home-stats').innerHTML = Components.emptyState('error', '加载统计数据失败');
            }
        }
    },

    // =========== 认证页面 ==========
    auth: {
        captchaId: null,
        
        // 登录页面
        async renderLogin() {
            document.body.classList.add('hide-header');
            
            const content = `
                <div class="auth-page">
                    <div class="auth-card">
                        <div class="auth-header">
                            <span class="auth-icon material-icons">rocket_launch</span>
                            <h1>AI-Tripod</h1>
                            <p>登录到您的账户</p>
                        </div>
                        <form class="auth-form" id="login-form">
                            <div class="form-group">
                                <label>用户名</label>
                                <input type="text" class="form-control" name="username" required placeholder="请输入用户名">
                            </div>
                            <div class="form-group">
                                <label>密码</label>
                                <input type="password" class="form-control" name="password" required placeholder="请输入密码">
                            </div>
                            <div class="form-group captcha-group">
                                <label>验证码</label>
                                <div class="captcha-input">
                                    <input type="text" class="form-control" name="captcha" required placeholder="请输入验证码">
                                    <img class="captcha-img" id="captcha-img" src="" alt="验证码" title="点击刷新" onclick="Pages.auth.refreshCaptcha()">
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary btn-block btn-lg">登录</button>
                        </form>
                        <div class="auth-footer">
                            还没有账户？<a href="#/register">立即注册</a>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.refreshCaptcha();
            
            // 绑定表单提交
            document.getElementById('login-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleLogin(e.target);
            });
        },
        
        // 注册页面
        async renderRegister() {
            document.body.classList.add('hide-header');
            
            const content = `
                <div class="auth-page">
                    <div class="auth-card">
                        <div class="auth-header">
                            <span class="auth-icon material-icons">rocket_launch</span>
                            <h1>AI-Tripod</h1>
                            <p>创建新账户</p>
                        </div>
                        <form class="auth-form" id="register-form">
                            <div class="form-group">
                                <label>用户名</label>
                                <input type="text" class="form-control" name="username" required placeholder="请输入用户名" minlength="3">
                            </div>
                            <div class="form-group">
                                <label>邮箱（可选）</label>
                                <input type="email" class="form-control" name="email" placeholder="请输入邮箱">
                            </div>
                            <div class="form-group">
                                <label>密码</label>
                                <input type="password" class="form-control" name="password" required placeholder="请输入密码" minlength="6">
                            </div>
                            <div class="form-group">
                                <label>确认密码</label>
                                <input type="password" class="form-control" name="confirmPassword" required placeholder="请再次输入密码">
                            </div>
                            <div class="form-group captcha-group">
                                <label>验证码</label>
                                <div class="captcha-input">
                                    <input type="text" class="form-control" name="captcha" required placeholder="请输入验证码">
                                    <img class="captcha-img" id="captcha-img" src="" alt="验证码" title="点击刷新" onclick="Pages.auth.refreshCaptcha()">
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary btn-block btn-lg">注册</button>
                        </form>
                        <div class="auth-footer">
                            已有账户？<a href="#/login">立即登录</a>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.refreshCaptcha();
            
            // 绑定表单提交
            document.getElementById('register-form').addEventListener('submit', async (e) => {
                e.preventDefault();
                await this.handleRegister(e.target);
            });
        },
        
        // 刷新验证码
        async refreshCaptcha() {
            const img = document.getElementById('captcha-img');
            if (img) {
                try {
                    // 从后端获取验证码并提取 captcha-id 响应头
                    const response = await fetch(`/api/captcha?t=${Date.now()}`);
                    const captchaId = response.headers.get('captcha-id');

                    if (captchaId) {
                        this.captchaId = captchaId;
                        // 使用获取到的图片 URL
                        const blob = await response.blob();
                        const imageUrl = URL.createObjectURL(blob);
                        img.src = imageUrl;
                    } else {
                        console.error('未能从响应头获取 captcha-id');
                        Utils.toast.error('获取验证码失败');
                    }
                } catch (error) {
                    console.error('刷新验证码失败:', error);
                    Utils.toast.error('刷新验证码失败');
                }
            }
        },
        
        // 处理登录
        async handleLogin(form) {
            const formData = new FormData(form);
            const submitBtn = form.querySelector('button[type="submit"]');
            
            try {
                submitBtn.disabled = true;
                submitBtn.textContent = '登录中...';
                
                await API.auth.login(
                    formData.get('username'),
                    formData.get('password'),
                    formData.get('captcha'),
                    this.captchaId
                );

                Utils.toast.success('登录成功');
                window.location.hash = '#/home';
            } catch (error) {
                Utils.toast.error(error.message);
                this.refreshCaptcha();
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '登录';
            }
        },
        
        // 处理注册
        async handleRegister(form) {
            const formData = new FormData(form);
            const submitBtn = form.querySelector('button[type="submit"]');
            
            // 验证密码
            if (formData.get('password') !== formData.get('confirmPassword')) {
                Utils.toast.error('两次输入的密码不一致');
                return;
            }
            
            try {
                submitBtn.disabled = true;
                submitBtn.textContent = '注册中...';
                
                await API.auth.register({
                    username: formData.get('username'),
                    email: formData.get('email') || undefined,
                    password: formData.get('password'),
                    captcha: formData.get('captcha'),
                    captcha_id: this.captchaId
                });
                
                Utils.toast.success('注册成功，请登录');
                window.location.hash = '#/login';
            } catch (error) {
                Utils.toast.error(error.message);
                this.refreshCaptcha();
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '注册';
            }
        }
    },
    
    // =========== 仪表盘页面 ===========
    dashboard: {
        async render() {
            document.body.classList.remove('hide-header');
            
            const content = `
                ${Components.pageHeader('仪表盘', '欢迎使用 AI-Tripod 模型管理平台')}
                
                <div class="stats-grid" id="stats-container">
                    ${Components.loading('加载统计数据...')}
                </div>
                
                <div class="dashboard-grid">
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">bolt</span> 快捷操作</h3>
                        </div>
                        <div class="card-body">
                            <div class="quick-actions">
                                <a href="#/resources" class="quick-action-btn">
                                    <span class="material-icons">cloud_download</span>
                                    <span>下载资源</span>
                                </a>
                                <a href="#/training" class="quick-action-btn">
                                    <span class="material-icons">model_training</span>
                                    <span>创建训练</span>
                                </a>
                                <a href="#/inference" class="quick-action-btn">
                                    <span class="material-icons">psychology</span>
                                    <span>启动推理</span>
                                </a>
                                <a href="#/evaluation" class="quick-action-btn">
                                    <span class="material-icons">analytics</span>
                                    <span>模型评估</span>
                                </a>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">memory</span> GPU 状态</h3>
                            <button class="btn btn-sm btn-outline" onclick="Pages.dashboard.refreshGpu()">
                                <span class="material-icons">refresh</span>
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="gpu-info" id="gpu-container">
                                ${Components.loading('加载 GPU 状态...')}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            
            // 加载数据
            this.loadStats();
            this.refreshGpu();
        },
        
        async loadStats() {
            try {
                // 并行获取统计数据
                const [resources, training, inference, evaluation] = await Promise.all([
                    API.resources.list(),
                    API.training.list(),
                    API.inference.list(),
                    API.evaluation.list()
                ]);
                
                const statsHtml = `
                    ${Components.statCard('folder', '资源总数', resources.length)}
                    ${Components.statCard('model_training', '训练任务', training.length)}
                    ${Components.statCard('psychology', '推理服务', inference.filter(t => t.status === 'RUNNING').length)}
                    ${Components.statCard('analytics', '评估任务', evaluation.length)}
                `;
                
                document.getElementById('stats-container').innerHTML = statsHtml;
            } catch (error) {
                document.getElementById('stats-container').innerHTML = Components.emptyState('error', '加载统计数据失败');
            }
        },
        
        async refreshGpu() {
            try {
                const gpuData = await API.inference.getGpuStatus();
                
                if (gpuData && gpuData.gpus && gpuData.gpus.length > 0) {
                    const gpuHtml = gpuData.gpus.map((gpu, index) => Components.gpuCard(gpu, index)).join('');
                    document.getElementById('gpu-container').innerHTML = gpuHtml;
                } else {
                    document.getElementById('gpu-container').innerHTML = Components.emptyState('memory', '未检测到 GPU');
                }
            } catch (error) {
                document.getElementById('gpu-container').innerHTML = Components.emptyState('error', 'GPU 状态获取失败');
            }
        }
    },
    
    // =========== 资源管理页面 ===========
    resources: {
        currentType: '',
        pollingTimer: null,
        
        async render() {
            document.body.classList.remove('hide-header');
            
            // 清除之前的轮询
            this.stopPolling();
            
            const toolbar = Components.toolbar(
                `<div class="filter-group">
                    <select class="form-control form-select" id="type-filter" style="width:auto" onchange="Pages.resources.filterByType(this.value)">
                        <option value="">全部类型</option>
                        <option value="MODEL">模型</option>
                        <option value="DATASET">数据集</option>
                    </select>
                </div>`,
                `<button class="btn btn-primary" onclick="Pages.resources.showCreateModal()">
                    <span class="material-icons">add</span> 添加资源
                </button>`
            );
            
            const content = `
                ${Components.pageHeader('资源管理', '管理您的模型和数据集资源')}
                ${toolbar}
                <div class="resources-grid" id="resources-container">
                    ${Components.loading('加载资源列表...')}
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.loadResources();
        },
        
        stopPolling() {
            if (this.pollingTimer) {
                clearInterval(this.pollingTimer);
                this.pollingTimer = null;
            }
        },
        
        startPolling() {
            this.stopPolling();
            // 每3秒刷新一次
            this.pollingTimer = setInterval(() => {
                this.loadResources(true); // true表示静默刷新，不显示loading
            }, 3000);
        },
        
        async loadResources(silent = false) {
            try {
                const params = this.currentType ? { type: this.currentType } : {};
                const resources = await API.resources.list(params);
                
                // 检查是否有正在下载的资源
                const hasDownloading = resources.some(r => r.status === 'DOWNLOADING');
                if (hasDownloading) {
                    this.startPolling();
                } else {
                    this.stopPolling();
                }
                
                if (resources.length === 0) {
                    document.getElementById('resources-container').innerHTML = Components.emptyState(
                        'folder_open',
                        '暂无资源',
                        '<button class="btn btn-primary" style="margin-top:var(--spacing-md)" onclick="Pages.resources.showCreateModal()">添加资源</button>'
                    );
                } else {
                    document.getElementById('resources-container').innerHTML = resources.map(r => Components.resourceCard(r)).join('');
                }
            } catch (error) {
                if (!silent) {
                    Utils.toast.error('加载资源失败: ' + error.message);
                }
            }
        },
        
        filterByType(type) {
            this.currentType = type;
            this.loadResources();
        },
        
        showCreateModal() {
            Utils.modal.form({
                title: '添加资源',
                fields: [
                    { name: 'name', label: '资源名称', required: true, placeholder: '例如: Qwen2-7B' },
                    { 
                        name: 'resource_type', 
                        label: '资源类型', 
                        type: 'select', 
                        required: true,
                        options: [
                            { value: 'MODEL', label: '模型' },
                            { value: 'DATASET', label: '数据集' }
                        ]
                    },
                    { name: 'repo_id', label: 'HuggingFace Repo ID', required: true, placeholder: '例如: Qwen/Qwen2-7B' },
                    { name: 'description', label: '描述', type: 'textarea', placeholder: '资源描述（可选）' }
                ],
                onSubmit: async (data) => {
                    await API.resources.create(data);
                    Utils.toast.success('资源创建成功');
                    this.loadResources();
                },
                submitText: '创建'
            });
        },
        
        async startDownload(id) {
            try {
                // 立即更新UI - 显示正在启动状态
                const card = document.querySelector(`.resource-card[data-id="${id}"]`);
                if (card) {
                    const actionsDiv = card.querySelector('.resource-actions');
                    if (actionsDiv) {
                        actionsDiv.innerHTML = `
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">hourglass_empty</span> 启动中...
                            </button>
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">info</span>
                            </button>
                            <button class="btn btn-sm btn-danger" disabled>
                                <span class="material-icons">delete</span>
                            </button>
                        `;
                    }
                }

                await API.resources.startDownload(id);
                Utils.toast.success('开始下载');
                this.loadResources();
            } catch (error) {
                Utils.toast.error(error.message);
                this.loadResources();
            }
        },

        async stopDownload(id) {
            try {
                // 立即更新UI - 显示正在停止状态
                const card = document.querySelector(`.resource-card[data-id="${id}"]`);
                if (card) {
                    const actionsDiv = card.querySelector('.resource-actions');
                    if (actionsDiv) {
                        actionsDiv.innerHTML = `
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">hourglass_empty</span> 停止中...
                            </button>
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">info</span>
                            </button>
                            <button class="btn btn-sm btn-danger" disabled>
                                <span class="material-icons">delete</span>
                            </button>
                        `;
                    }
                }

                await API.resources.stopDownload(id);
                Utils.toast.success('已停止下载');

                // 立即刷新列表
                this.loadResources();
            } catch (error) {
                Utils.toast.error(error.message);
                // 出错时也要刷新列表以恢复状态
                this.loadResources();
            }
        },

        async retryDownload(id) {
            try {
                // 立即更新UI - 显示正在重试状态
                const card = document.querySelector(`.resource-card[data-id="${id}"]`);
                if (card) {
                    const actionsDiv = card.querySelector('.resource-actions');
                    if (actionsDiv) {
                        actionsDiv.innerHTML = `
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">hourglass_empty</span> 重试中...
                            </button>
                            <button class="btn btn-sm btn-outline" disabled>
                                <span class="material-icons">info</span>
                            </button>
                            <button class="btn btn-sm btn-danger" disabled>
                                <span class="material-icons">delete</span>
                            </button>
                        `;
                    }
                }

                await API.resources.retryDownload(id);
                Utils.toast.success('重新开始下载');
                this.loadResources();
            } catch (error) {
                Utils.toast.error(error.message);
                this.loadResources();
            }
        },
        
        async deleteResource(id) {
            Utils.modal.confirm('确定要删除这个资源吗？此操作不可恢复。', async () => {
                try {
                    await API.resources.delete(id);
                    Utils.toast.success('资源已删除');
                    this.loadResources();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        },
        
        async showDetail(id) {
            try {
                const resource = await API.resources.get(id);
                const resType = (resource.resource_type || resource.type || '').toUpperCase();
                const typeText = resType === 'MODEL' ? '模型' : '数据集';
                const statusInfo = Utils.status.resourceStatus[resource.status] || { text: resource.status };
                
                Utils.modal.open({
                    title: resource.name,
                    content: `
                        <div class="info-grid">
                            ${Components.infoItem('类型', typeText)}
                            ${Components.infoItem('状态', statusInfo.text)}
                            ${Components.infoItem('Repo ID', resource.repo_id)}
                            ${Components.infoItem('本地路径', resource.local_path || '-')}
                            ${Components.infoItem('创建时间', Utils.format.date(resource.created_at))}
                            ${Components.infoItem('更新时间', Utils.format.date(resource.updated_at))}
                        </div>
                        ${resource.description ? `<p style="margin-top:var(--spacing-md);color:var(--text-secondary)">${resource.description}</p>` : ''}
                    `,
                    footer: '<button class="btn btn-secondary" onclick="Utils.modal.close()">关闭</button>',
                    size: 'medium'
                });
            } catch (error) {
                Utils.toast.error(error.message);
            }
        }
    },
    
    // =========== 训练任务页面 ===========
    training: {
        pollingTimer: null,

        async render() {
            document.body.classList.remove('hide-header');

            const toolbar = Components.toolbar(
                '',
                `<button class="btn btn-primary" onclick="Pages.training.showCreateModal()">
                    <span class="material-icons">add</span> 创建训练任务
                </button>`
            );

            const content = `
                ${Components.pageHeader('训练任务', '管理模型训练任务')}
                ${toolbar}
                <div class="tasks-grid" id="training-container">
                    ${Components.loading('加载训练任务...')}
                </div>
            `;

            document.getElementById('main-content').innerHTML = content;
            this.loadTasks();
            this.startPolling();
        },

        stopPolling() {
            if (this.pollingTimer) {
                clearInterval(this.pollingTimer);
                this.pollingTimer = null;
            }
        },

        startPolling() {
            this.stopPolling();
            // 每5秒刷新一次
            this.pollingTimer = setInterval(() => {
                this.loadTasks(true);
            }, 5000);
        },
        
        async loadTasks(silent = false) {
            try {
                const tasks = await API.training.list();

                if (tasks.length === 0) {
                    document.getElementById('training-container').innerHTML = Components.emptyState(
                        'model_training',
                        '暂无训练任务',
                        '<button class="btn btn-primary" style="margin-top:var(--spacing-md)" onclick="Pages.training.showCreateModal()">创建任务</button>'
                    );
                } else {
                    document.getElementById('training-container').innerHTML = tasks.map(t => Components.trainingCard(t)).join('');
                }
            } catch (error) {
                if (!silent) {
                    Utils.toast.error('加载任务失败: ' + error.message);
                }
            }
        },
        
        async showCreateModal() {
            // 先获取可用模型
            let models = [];
            let datasets = [];
            try {
                const resources = await API.resources.list();
                models = resources.filter(r => (r.resource_type || r.type) === 'MODEL' && r.status === 'COMPLETED');
                datasets = resources.filter(r => (r.resource_type || r.type) === 'DATASET' && r.status === 'COMPLETED');
            } catch (error) {
                Utils.toast.error('获取资源列表失败');
                return;
            }
            
            if (models.length === 0) {
                Utils.toast.warning('请先下载至少一个模型');
                return;
            }
            
            Utils.modal.form({
                title: '创建训练任务',
                size: 'large',
                fields: [
                    { name: 'name', label: '任务名称', required: true, placeholder: '例如: SFT 训练' },
                    { 
                        name: 'model_id', 
                        label: '基础模型', 
                        type: 'select', 
                        required: true,
                        options: models.map(m => ({ value: m.id, label: m.name }))
                    },
                    { 
                        name: 'dataset_id', 
                        label: '训练数据集', 
                        type: 'select', 
                        required: false,
                        options: [{ value: '', label: '无' }, ...datasets.map(d => ({ value: d.id, label: d.name }))]
                    },
                    { name: 'epochs', label: '训练轮数', type: 'number', value: '3', placeholder: '默认 3' },
                    { name: 'batch_size', label: 'Batch Size', type: 'number', value: '4', placeholder: '默认 4' },
                    { name: 'learning_rate', label: '学习率', value: '2e-5', placeholder: '默认 2e-5' }
                ],
                onSubmit: async (data) => {
                    const createData = {
                        name: data.name,
                        base_model_id: parseInt(data.model_id),
                        dataset_id: data.dataset_id ? parseInt(data.dataset_id) : null,
                        config_params: {
                            epochs: parseInt(data.epochs) || 3,
                            batch_size: parseInt(data.batch_size) || 4,
                            learning_rate: parseFloat(data.learning_rate) || 2e-5
                        }
                    };
                    await API.training.create(createData);
                    Utils.toast.success('训练任务创建成功');
                    this.loadTasks();
                },
                submitText: '创建'
            });
        },
        
        async startTask(id) {
            try {
                await API.training.start(id);
                Utils.toast.success('训练任务已启动');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async stopTask(id) {
            try {
                await API.training.stop(id);
                Utils.toast.success('训练任务已停止');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async deleteTask(id) {
            Utils.modal.confirm('确定要删除这个训练任务吗？', async () => {
                try {
                    await API.training.delete(id);
                    Utils.toast.success('任务已删除');
                    this.loadTasks();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        },
        
        // 训练详情页
        async renderDetail(params) {
            document.body.classList.remove('hide-header');
            
            const content = `
                ${Components.pageHeader('训练详情')}
                <a href="#/training" class="back-link">
                    <span class="material-icons">arrow_back</span> 返回列表
                </a>
                <div class="detail-grid" id="training-detail">
                    ${Components.loading('加载任务详情...')}
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.loadDetail(params.id);
        },
        
        async loadDetail(id) {
            try {
                const task = await API.training.get(id);
                const statusBadge = Utils.status.getBadge(task.status, 'training');
                
                let actionsHtml = '';
                if (task.status === 'PENDING') {
                    actionsHtml = `<button class="btn btn-success" onclick="Pages.training.startTask(${task.id}); Pages.training.loadDetail(${task.id});">启动训练</button>`;
                } else if (task.status === 'RUNNING') {
                    actionsHtml = `<button class="btn btn-warning" onclick="Pages.training.stopTask(${task.id}); Pages.training.loadDetail(${task.id});">停止训练</button>`;
                }
                
                const detailHtml = `
                    <div class="card">
                        <div class="card-header">
                            <h3>${task.name} ${statusBadge}</h3>
                        </div>
                        <div class="card-body">
                            <div class="info-grid">
                                ${Components.infoItem('任务 ID', task.id)}
                                ${Components.infoItem('模型', task.model_name || '-')}
                                ${Components.infoItem('数据集', task.dataset_name || '-')}
                                ${Components.infoItem('状态', Utils.status.trainingStatus[task.status]?.text || task.status)}
                                ${Components.infoItem('创建时间', Utils.format.date(task.created_at))}
                                ${Components.infoItem('开始时间', Utils.format.date(task.started_at))}
                                ${Components.infoItem('完成时间', Utils.format.date(task.completed_at))}
                                ${Components.infoItem('输出路径', task.output_path || '-')}
                            </div>
                            <div class="action-buttons">
                                ${actionsHtml}
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">terminal</span> 训练日志</h3>
                            <button class="btn btn-sm btn-outline" onclick="Pages.training.refreshLogs(${task.id})">
                                <span class="material-icons">refresh</span>
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="log-container" id="log-container">
                                ${Components.loading('加载日志...')}
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('training-detail').innerHTML = detailHtml;
                this.refreshLogs(id);
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async refreshLogs(id) {
            try {
                const logs = await API.training.getLogs(id);
                const logContainer = document.getElementById('log-container');
                
                if (logs && logs.length > 0) {
                    logContainer.innerHTML = logs.map(log => Components.logEntry(log)).join('');
                    logContainer.scrollTop = logContainer.scrollHeight;
                } else {
                    logContainer.innerHTML = '<p style="color:var(--text-tertiary)">暂无日志</p>';
                }
            } catch (error) {
                document.getElementById('log-container').innerHTML = '<p style="color:var(--text-tertiary)">加载日志失败</p>';
            }
        }
    },
    
    // =========== 推理服务页面 ===========
    inference: {
        pollingTimer: null,

        async render() {
            document.body.classList.remove('hide-header');

            const toolbar = Components.toolbar(
                '',
                `<button class="btn btn-primary" onclick="Pages.inference.showCreateModal()">
                    <span class="material-icons">add</span> 创建推理服务
                </button>`
            );

            const content = `
                ${Components.pageHeader('推理服务', '管理模型推理服务')}
                ${toolbar}
                <div class="tasks-grid" id="inference-container">
                    ${Components.loading('加载推理服务...')}
                </div>
            `;

            document.getElementById('main-content').innerHTML = content;
            this.loadTasks();
            this.startPolling();
        },

        stopPolling() {
            if (this.pollingTimer) {
                clearInterval(this.pollingTimer);
                this.pollingTimer = null;
            }
        },

        startPolling() {
            this.stopPolling();
            // 每5秒刷新一次
            this.pollingTimer = setInterval(() => {
                this.loadTasks(true);
            }, 5000);
        },

        async loadTasks(silent = false) {
            try {
                const tasks = await API.inference.list();

                if (tasks.length === 0) {
                    document.getElementById('inference-container').innerHTML = Components.emptyState(
                        'psychology',
                        '暂无推理服务',
                        '<button class="btn btn-primary" style="margin-top:var(--spacing-md)" onclick="Pages.inference.showCreateModal()">创建服务</button>'
                    );
                } else {
                    document.getElementById('inference-container').innerHTML = tasks.map(t => Components.inferenceCard(t)).join('');
                }
            } catch (error) {
                if (!silent) {
                    Utils.toast.error('加载服务失败: ' + error.message);
                }
            }
        },
        
        async showCreateModal() {
            // 获取可用模型
            let models = [];
            try {
                const resources = await API.resources.list();
                models = resources.filter(r => (r.resource_type || r.type) === 'MODEL' && r.status === 'COMPLETED');
            } catch (error) {
                Utils.toast.error('获取模型列表失败');
                return;
            }
            
            if (models.length === 0) {
                Utils.toast.warning('请先下载至少一个模型');
                return;
            }
            
            Utils.modal.form({
                title: '创建推理服务',
                fields: [
                    { name: 'name', label: '服务名称', required: true, placeholder: '例如: Chat 服务' },
                    { 
                        name: 'model_id', 
                        label: '模型', 
                        type: 'select', 
                        required: true,
                        options: models.map(m => ({ value: m.id, label: m.name }))
                    },
                    { name: 'port', label: '端口', type: 'number', value: '8000', placeholder: '默认 8000' },
                    { name: 'gpu_memory_utilization', label: 'GPU 显存利用率', value: '0.9', placeholder: '0.0-1.0' }
                ],
                onSubmit: async (data) => {
                    await API.inference.create({
                        name: data.name,
                        model_id: parseInt(data.model_id),
                        port: parseInt(data.port) || 8000,
                        gpu_memory_utilization: parseFloat(data.gpu_memory_utilization) || 0.9
                    });
                    Utils.toast.success('推理服务创建成功');
                    this.loadTasks();
                },
                submitText: '创建'
            });
        },
        
        async startTask(id) {
            try {
                await API.inference.start(id);
                Utils.toast.success('推理服务启动中...');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async stopTask(id) {
            try {
                await API.inference.stop(id);
                Utils.toast.success('推理服务已停止');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async deleteTask(id) {
            Utils.modal.confirm('确定要删除这个推理服务吗？', async () => {
                try {
                    await API.inference.delete(id);
                    Utils.toast.success('服务已删除');
                    this.loadTasks();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        }
    },
    
    // =========== 聊天页面 ===========
    chat: {
        ws: null,
        messages: [],
        taskId: null,
        
        async render(params) {
            document.body.classList.remove('hide-header');
            this.taskId = params.id;
            this.messages = [];
            
            // 获取服务信息
            let task;
            try {
                task = await API.inference.get(params.id);
            } catch (error) {
                Utils.toast.error('获取服务信息失败');
                window.location.hash = '#/inference';
                return;
            }
            
            if (task.status !== 'RUNNING') {
                Utils.toast.warning('推理服务未运行');
                window.location.hash = '#/inference';
                return;
            }
            
            const content = `
                <div class="chat-page">
                    <div class="chat-header">
                        <a href="#/inference" class="btn btn-outline">
                            <span class="material-icons">arrow_back</span>
                        </a>
                        <div class="chat-info">
                            <h2>${task.name}</h2>
                            <span style="color:var(--text-secondary);font-size:0.875rem">${task.model_name || '模型对话'}</span>
                        </div>
                        <button class="btn btn-outline" onclick="Pages.chat.clearMessages()">
                            <span class="material-icons">delete_sweep</span> 清空
                        </button>
                    </div>
                    
                    <div class="chat-container">
                        <div class="messages-container" id="messages-container">
                            <div class="welcome-message">
                                <span class="material-icons">smart_toy</span>
                                <h3>开始对话</h3>
                                <p>输入您的问题，按 Enter 发送</p>
                            </div>
                        </div>
                        
                        <div class="input-container">
                            <div class="input-wrapper">
                                <textarea id="chat-input" placeholder="输入消息..." rows="1" onkeydown="Pages.chat.handleKeyDown(event)"></textarea>
                                <button class="send-btn" onclick="Pages.chat.sendMessage()">
                                    <span class="material-icons">send</span>
                                </button>
                            </div>
                            <p class="input-hint">按 Enter 发送，Shift + Enter 换行</p>
                        </div>
                    </div>
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            document.getElementById('chat-input').focus();
        },
        
        handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                this.sendMessage();
            }
        },
        
        async sendMessage() {
            const input = document.getElementById('chat-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // 添加用户消息
            this.addMessage('user', message);
            input.value = '';
            
            // 发送请求
            try {
                const response = await API.inference.chat(this.taskId, {
                    messages: this.messages.map(m => ({
                        role: m.role,
                        content: m.content
                    }))
                });
                
                // 添加助手回复
                const assistantContent =
                    (response && response.message && response.message.content) ||
                    (response && response.content) ||
                    (response && response.choices && response.choices[0] && response.choices[0].message && response.choices[0].message.content);

                if (assistantContent) {
                    this.addMessage('assistant', assistantContent);
                } else {
                    Utils.toast.error('未获取到模型回复（响应格式不匹配）');
                }
            } catch (error) {
                Utils.toast.error('发送失败: ' + error.message);
                // 移除用户消息
                this.messages.pop();
                this.renderMessages();
            }
        },
        
        addMessage(role, content) {
            this.messages.push({ role, content });
            this.renderMessages();
        },
        
        renderMessages() {
            const container = document.getElementById('messages-container');
            
            if (this.messages.length === 0) {
                container.innerHTML = `
                    <div class="welcome-message">
                        <span class="material-icons">smart_toy</span>
                        <h3>开始对话</h3>
                        <p>输入您的问题，按 Enter 发送</p>
                    </div>
                `;
            } else {
                container.innerHTML = this.messages.map(m => Components.chatMessage(m.content, m.role)).join('');
                container.scrollTop = container.scrollHeight;
            }
        },
        
        clearMessages() {
            this.messages = [];
            this.renderMessages();
        }
    },
    
    // =========== 评估任务页面 ===========
    evaluation: {
        pollingTimer: null,

        async render() {
            document.body.classList.remove('hide-header');

            const toolbar = Components.toolbar(
                '',
                `<button class="btn btn-primary" onclick="Pages.evaluation.showCreateModal()">
                    <span class="material-icons">add</span> 创建评估任务
                </button>`
            );

            const content = `
                ${Components.pageHeader('模型评估', '评估模型性能')}
                ${toolbar}
                <div class="tasks-grid" id="evaluation-container">
                    ${Components.loading('加载评估任务...')}
                </div>
            `;

            document.getElementById('main-content').innerHTML = content;
            this.loadTasks();
            this.startPolling();
        },

        stopPolling() {
            if (this.pollingTimer) {
                clearInterval(this.pollingTimer);
                this.pollingTimer = null;
            }
        },

        startPolling() {
            this.stopPolling();
            // 每5秒刷新一次
            this.pollingTimer = setInterval(() => {
                this.loadTasks(true);
            }, 5000);
        },

        async loadTasks(silent = false) {
            try {
                const tasks = await API.evaluation.list();

                if (tasks.length === 0) {
                    document.getElementById('evaluation-container').innerHTML = Components.emptyState(
                        'analytics',
                        '暂无评估任务',
                        '<button class="btn btn-primary" style="margin-top:var(--spacing-md)" onclick="Pages.evaluation.showCreateModal()">创建任务</button>'
                    );
                } else {
                    document.getElementById('evaluation-container').innerHTML = tasks.map(t => Components.evaluationCard(t)).join('');
                }
            } catch (error) {
                if (!silent) {
                    Utils.toast.error('加载任务失败: ' + error.message);
                }
            }
        },
        
        async showCreateModal() {
            // 获取可用模型
            let models = [];
            try {
                const resources = await API.resources.list();
                models = resources.filter(r => (r.resource_type || r.type) === 'MODEL' && r.status === 'COMPLETED');
            } catch (error) {
                Utils.toast.error('获取模型列表失败');
                return;
            }
            
            if (models.length === 0) {
                Utils.toast.warning('请先下载至少一个模型');
                return;
            }
            
            Utils.modal.form({
                title: '创建评估任务',
                fields: [
                    { name: 'name', label: '任务名称', required: true, placeholder: '例如: MMLU 评估' },
                    { 
                        name: 'model_id', 
                        label: '模型', 
                        type: 'select', 
                        required: true,
                        options: models.map(m => ({ value: m.id, label: m.name }))
                    },
                    { 
                        name: 'benchmark', 
                        label: '评估基准', 
                        type: 'select', 
                        required: true,
                        options: [
                            { value: 'mmlu', label: 'MMLU (大规模多任务语言理解)' }
                        ]
                    },
                    { name: 'num_fewshot', label: 'Few-shot 数量', type: 'number', value: '5', placeholder: '默认 5' }
                ],
                onSubmit: async (data) => {
                    await API.evaluation.create({
                        name: data.name,
                        model_id: parseInt(data.model_id),
                        benchmark_type: data.benchmark,
                        num_fewshot: parseInt(data.num_fewshot) || 5
                    });
                    Utils.toast.success('评估任务创建成功');
                    this.loadTasks();
                },
                submitText: '创建'
            });
        },
        
        async startTask(id) {
            try {
                await API.evaluation.start(id);
                Utils.toast.success('评估任务已启动');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async stopTask(id) {
            try {
                await API.evaluation.stop(id);
                Utils.toast.success('评估任务已停止');
                this.loadTasks();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async deleteTask(id) {
            Utils.modal.confirm('确定要删除这个评估任务吗？', async () => {
                try {
                    await API.evaluation.delete(id);
                    Utils.toast.success('任务已删除');
                    this.loadTasks();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        },
        
        // 评估详情页
        async renderDetail(params) {
            document.body.classList.remove('hide-header');
            
            const content = `
                ${Components.pageHeader('评估详情')}
                <a href="#/evaluation" class="back-link">
                    <span class="material-icons">arrow_back</span> 返回列表
                </a>
                <div class="detail-grid" id="evaluation-detail">
                    ${Components.loading('加载任务详情...')}
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.loadDetail(params.id);
        },
        
        async loadDetail(id) {
            try {
                const task = await API.evaluation.get(id);
                const statusBadge = Utils.status.getBadge(task.status, 'evaluation');
                
                let metricsHtml = '';
                if (task.metrics) {
                    const metrics = typeof task.metrics === 'string' ? JSON.parse(task.metrics) : task.metrics;
                    metricsHtml = `
                        <div class="card" style="margin-top:var(--spacing-lg)">
                            <div class="card-header">
                                <h3><span class="material-icons">insights</span> 评估结果</h3>
                            </div>
                            <div class="card-body">
                                <div class="info-grid">
                                    ${Object.entries(metrics).map(([key, value]) => 
                                        Components.infoItem(key, typeof value === 'number' ? (value * 100).toFixed(2) + '%' : value)
                                    ).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                }
                
                const detailHtml = `
                    <div class="card">
                        <div class="card-header">
                            <h3>${task.name} ${statusBadge}</h3>
                        </div>
                        <div class="card-body">
                            <div class="info-grid">
                                ${Components.infoItem('任务 ID', task.id)}
                                ${Components.infoItem('模型', task.model_name || '-')}
                                ${Components.infoItem('评估基准', task.benchmark)}
                                ${Components.infoItem('状态', Utils.status.evaluationStatus[task.status]?.text || task.status)}
                                ${Components.infoItem('创建时间', Utils.format.date(task.created_at))}
                                ${Components.infoItem('完成时间', Utils.format.date(task.completed_at))}
                            </div>
                        </div>
                    </div>
                    ${metricsHtml}
                    
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">terminal</span> 评估日志</h3>
                            <button class="btn btn-sm btn-outline" onclick="Pages.evaluation.refreshLogs(${task.id})">
                                <span class="material-icons">refresh</span>
                            </button>
                        </div>
                        <div class="card-body">
                            <div class="log-container" id="log-container">
                                ${Components.loading('加载日志...')}
                            </div>
                        </div>
                    </div>
                `;
                
                document.getElementById('evaluation-detail').innerHTML = detailHtml;
                this.refreshLogs(id);
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async refreshLogs(id) {
            try {
                const logs = await API.evaluation.getLogs(id);
                const logContainer = document.getElementById('log-container');
                
                if (logs && logs.length > 0) {
                    logContainer.innerHTML = logs.map(log => Components.logEntry(log)).join('');
                    logContainer.scrollTop = logContainer.scrollHeight;
                } else {
                    logContainer.innerHTML = '<p style="color:var(--text-tertiary)">暂无日志</p>';
                }
            } catch (error) {
                document.getElementById('log-container').innerHTML = '<p style="color:var(--text-tertiary)">加载日志失败</p>';
            }
        }
    },
    
    // =========== 个人资料页面 ===========
    profile: {
        async render() {
            document.body.classList.remove('hide-header');
            
            const content = `
                ${Components.pageHeader('个人设置', '管理您的账户信息')}
                <div class="profile-grid" id="profile-container">
                    ${Components.loading('加载个人资料...')}
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.loadProfile();
        },
        
        async loadProfile() {
            try {
                const user = await API.auth.getCurrentUser();
                
                const profileHtml = `
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">person</span> 基本信息</h3>
                        </div>
                        <div class="card-body">
                            <div class="info-grid">
                                ${Components.infoItem('用户名', user.username)}
                                ${Components.infoItem('邮箱', user.email || '未设置')}
                                ${Components.infoItem('角色', user.is_admin ? '管理员' : '普通用户')}
                                ${Components.infoItem('注册时间', Utils.format.date(user.created_at))}
                            </div>
                        </div>
                    </div>
                    
                    <div class="card">
                        <div class="card-header">
                            <h3><span class="material-icons">lock</span> 修改密码</h3>
                        </div>
                        <div class="card-body">
                            <form id="password-form">
                                <div class="form-group">
                                    <label>当前密码</label>
                                    <input type="password" class="form-control" name="old_password" required>
                                </div>
                                <div class="form-group">
                                    <label>新密码</label>
                                    <input type="password" class="form-control" name="new_password" required minlength="6">
                                </div>
                                <div class="form-group">
                                    <label>确认新密码</label>
                                    <input type="password" class="form-control" name="confirm_password" required>
                                </div>
                                <button type="submit" class="btn btn-primary">修改密码</button>
                            </form>
                        </div>
                    </div>
                `;
                
                document.getElementById('profile-container').innerHTML = profileHtml;
                
                // 绑定表单提交
                document.getElementById('password-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    await this.changePassword(e.target);
                });
            } catch (error) {
                Utils.toast.error(error.message);
            }
        },
        
        async changePassword(form) {
            const formData = new FormData(form);
            
            if (formData.get('new_password') !== formData.get('confirm_password')) {
                Utils.toast.error('两次输入的新密码不一致');
                return;
            }
            
            try {
                await API.profile.changePassword({
                    old_password: formData.get('old_password'),
                    new_password: formData.get('new_password')
                });
                Utils.toast.success('密码修改成功');
                form.reset();
            } catch (error) {
                Utils.toast.error(error.message);
            }
        }
    },
    
    // =========== 管理员页面 ===========
    admin: {
        async render() {
            document.body.classList.remove('hide-header');
            
            const toolbar = Components.toolbar(
                '',
                `<button class="btn btn-primary" onclick="Pages.admin.showCreateModal()">
                    <span class="material-icons">person_add</span> 添加用户
                </button>`
            );
            
            const content = `
                ${Components.pageHeader('用户管理', '管理系统用户')}
                ${toolbar}
                <div class="card">
                    <div class="card-body" style="padding:0;overflow-x:auto">
                        <table class="data-table" id="users-table">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>用户名</th>
                                    <th>邮箱</th>
                                    <th>管理员</th>
                                    <th>创建时间</th>
                                    <th>操作</th>
                                </tr>
                            </thead>
                            <tbody id="users-container">
                                <tr><td colspan="6">${Components.loading('加载用户列表...')}</td></tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
            
            document.getElementById('main-content').innerHTML = content;
            this.loadUsers();
        },
        
        async loadUsers() {
            try {
                const users = await API.users.list();
                
                if (users.length === 0) {
                    document.getElementById('users-container').innerHTML = '<tr><td colspan="6" style="text-align:center">暂无用户</td></tr>';
                } else {
                    document.getElementById('users-container').innerHTML = users.map(u => Components.userCard(u)).join('');
                }
            } catch (error) {
                Utils.toast.error('加载用户失败: ' + error.message);
            }
        },
        
        showCreateModal() {
            Utils.modal.form({
                title: '添加用户',
                fields: [
                    { name: 'username', label: '用户名', required: true, placeholder: '请输入用户名' },
                    { name: 'email', label: '邮箱', type: 'email', placeholder: '请输入邮箱（可选）' },
                    { name: 'password', label: '密码', type: 'password', required: true, placeholder: '请输入密码' },
                    { 
                        name: 'is_admin', 
                        label: '设为管理员', 
                        type: 'select',
                        options: [
                            { value: 'false', label: '否' },
                            { value: 'true', label: '是' }
                        ]
                    }
                ],
                onSubmit: async (data) => {
                    await API.users.create({
                        username: data.username,
                        email: data.email || undefined,
                        password: data.password,
                        is_admin: data.is_admin === 'true'
                    });
                    Utils.toast.success('用户创建成功');
                    this.loadUsers();
                },
                submitText: '创建'
            });
        },
        
        async deleteUser(id) {
            Utils.modal.confirm('确定要删除这个用户吗？', async () => {
                try {
                    await API.users.delete(id);
                    Utils.toast.success('用户已删除');
                    this.loadUsers();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        }
    }
};

// 导出
window.Pages = Pages;
