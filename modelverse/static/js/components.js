/**
 * AI-Tripod 组件库
 * 可复用的 UI 组件
 */

const Components = {
    // =========== 页面头部 ===========
    pageHeader(title, subtitle = '') {
        return `
            <div class="page-header">
                <h1>${title}</h1>
                ${subtitle ? `<p class="subtitle">${subtitle}</p>` : ''}
            </div>
        `;
    },
    
    // =========== 统计卡片 ===========
    statCard(icon, label, value, color = '') {
        return `
            <div class="stat-card">
                <div class="stat-icon" ${color ? `style="color:${color}"` : ''}>
                    <span class="material-icons">${icon}</span>
                </div>
                <div class="stat-info">
                    <div class="stat-value">${value}</div>
                    <div class="stat-label">${label}</div>
                </div>
            </div>
        `;
    },
    
    // =========== 资源卡片 ===========
    resourceCard(resource) {
        const resType = (resource.resource_type || resource.type || '').toUpperCase();
        const typeClass = resType === 'MODEL' ? 'type-model' : 'type-dataset';
        const typeText = resType === 'MODEL' ? '模型' : '数据集';
        const status = (resource.status || '').toUpperCase();
        const statusBadge = Utils.status.getBadge(status, 'resource');
        
        let actionsHtml = '';
        
        if (status === 'PENDING') {
            actionsHtml = `
                <button class="btn btn-sm btn-primary" onclick="Pages.resources.startDownload(${resource.id})">
                    <span class="material-icons">download</span> 下载
                </button>
            `;
        } else if (status === 'DOWNLOADING') {
            actionsHtml = `
                <button class="btn btn-sm btn-warning" onclick="Pages.resources.stopDownload(${resource.id})">
                    <span class="material-icons">stop</span> 停止
                </button>
            `;
        } else if (status === 'FAILED') {
            actionsHtml = `
                <button class="btn btn-sm btn-primary" onclick="Pages.resources.retryDownload(${resource.id})">
                    <span class="material-icons">refresh</span> 重试
                </button>
            `;
        }
        
        actionsHtml += `
            <button class="btn btn-sm btn-outline" onclick="Pages.resources.showDetail(${resource.id})">
                <span class="material-icons">info</span>
            </button>
            <button class="btn btn-sm btn-danger" onclick="Pages.resources.deleteResource(${resource.id})">
                <span class="material-icons">delete</span>
            </button>
        `;
        
        let progressHtml = '';
        if (status === 'DOWNLOADING' && resource.progress !== undefined) {
            progressHtml = `
                <div class="progress-section">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${resource.progress}%"></div>
                    </div>
                    <span class="progress-text">${resource.progress.toFixed(1)}%</span>
                </div>
            `;
        }
        
        return `
            <div class="resource-card" data-id="${resource.id}">
                <div class="resource-header">
                    <div>
                        <h4 class="resource-name">${resource.name}</h4>
                        <span class="resource-type ${typeClass}">${typeText}</span>
                    </div>
                    ${statusBadge}
                </div>
                <div class="resource-info">
                    <code class="resource-repo">${resource.repo_id}</code>
                    ${resource.description ? `<p class="resource-description">${resource.description}</p>` : ''}
                </div>
                ${progressHtml}
                <div class="resource-actions">
                    ${actionsHtml}
                </div>
            </div>
        `;
    },
    
    // =========== 训练任务卡片 ===========
    trainingCard(task) {
        const statusBadge = Utils.status.getBadge(task.status, 'training');
        
        let actionsHtml = '';
        
        if (task.status === 'PENDING') {
            actionsHtml = `
                <button class="btn btn-sm btn-success" onclick="Pages.training.startTask(${task.id})">
                    <span class="material-icons">play_arrow</span> 启动
                </button>
            `;
        } else if (task.status === 'RUNNING') {
            actionsHtml = `
                <button class="btn btn-sm btn-warning" onclick="Pages.training.stopTask(${task.id})">
                    <span class="material-icons">stop</span> 停止
                </button>
            `;
        }
        
        actionsHtml += `
            <a href="#/training/${task.id}" class="btn btn-sm btn-outline">
                <span class="material-icons">visibility</span> 详情
            </a>
            <button class="btn btn-sm btn-danger" onclick="Pages.training.deleteTask(${task.id})">
                <span class="material-icons">delete</span>
            </button>
        `;
        
        let progressHtml = '';
        if (task.status === 'RUNNING' && task.progress !== undefined) {
            progressHtml = `
                <div class="progress-section">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${task.progress}%"></div>
                    </div>
                    <span class="progress-text">${task.progress.toFixed(1)}%</span>
                </div>
            `;
        }
        
        return `
            <div class="task-card" data-id="${task.id}">
                <div class="task-header">
                    <div>
                        <h4 class="task-name">${task.name}</h4>
                    </div>
                    ${statusBadge}
                </div>
                <div class="task-info">
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        模型: ${task.model_name || '-'}
                    </p>
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        创建: ${Utils.format.relativeTime(task.created_at)}
                    </p>
                </div>
                ${progressHtml}
                <div class="task-actions">
                    ${actionsHtml}
                </div>
            </div>
        `;
    },
    
    // =========== 推理任务卡片 ===========
    inferenceCard(task) {
        const statusBadge = Utils.status.getBadge(task.status, 'inference');
        
        let actionsHtml = '';
        
        if (task.status === 'STOPPED' || task.status === 'FAILED') {
            actionsHtml = `
                <button class="btn btn-sm btn-success" onclick="Pages.inference.startTask(${task.id})">
                    <span class="material-icons">play_arrow</span> 启动
                </button>
            `;
        } else if (task.status === 'RUNNING') {
            actionsHtml = `
                <a href="#/chat/${task.id}" class="btn btn-sm btn-primary">
                    <span class="material-icons">chat</span> 对话
                </a>
                <button class="btn btn-sm btn-warning" onclick="Pages.inference.stopTask(${task.id})">
                    <span class="material-icons">stop</span> 停止
                </button>
            `;
        } else if (task.status === 'CREATING') {
            actionsHtml = `
                <button class="btn btn-sm btn-secondary" disabled>
                    <span class="material-icons">hourglass_empty</span> 启动中...
                </button>
            `;
        }
        
        actionsHtml += `
            <button class="btn btn-sm btn-danger" onclick="Pages.inference.deleteTask(${task.id})">
                <span class="material-icons">delete</span>
            </button>
        `;
        
        return `
            <div class="task-card" data-id="${task.id}">
                <div class="task-header">
                    <div>
                        <h4 class="task-name">${task.name}</h4>
                    </div>
                    ${statusBadge}
                </div>
                <div class="task-info">
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        模型: ${task.model_name || '-'}
                    </p>
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        端口: ${task.port || '-'}
                    </p>
                </div>
                <div class="task-actions">
                    ${actionsHtml}
                </div>
            </div>
        `;
    },
    
    // =========== 评估任务卡片 ===========
    evaluationCard(task) {
        const statusBadge = Utils.status.getBadge(task.status, 'evaluation');
        
        let actionsHtml = '';
        
        if (task.status === 'PENDING') {
            actionsHtml = `
                <button class="btn btn-sm btn-success" onclick="Pages.evaluation.startTask(${task.id})">
                    <span class="material-icons">play_arrow</span> 启动
                </button>
            `;
        } else if (task.status === 'RUNNING') {
            actionsHtml = `
                <button class="btn btn-sm btn-warning" onclick="Pages.evaluation.stopTask(${task.id})">
                    <span class="material-icons">stop</span> 停止
                </button>
            `;
        }
        
        actionsHtml += `
            <a href="#/evaluation/${task.id}" class="btn btn-sm btn-outline">
                <span class="material-icons">visibility</span> 详情
            </a>
            <button class="btn btn-sm btn-danger" onclick="Pages.evaluation.deleteTask(${task.id})">
                <span class="material-icons">delete</span>
            </button>
        `;
        
        // 显示评估结果
        let metricsHtml = '';
        if (task.status === 'COMPLETED' && task.metrics) {
            const metrics = typeof task.metrics === 'string' ? JSON.parse(task.metrics) : task.metrics;
            if (metrics.accuracy !== undefined) {
                metricsHtml = `
                    <div style="margin-top:var(--spacing-sm);padding:var(--spacing-sm);background:var(--bg-secondary);border-radius:var(--border-radius)">
                        <span style="font-size:0.8125rem;color:var(--text-secondary)">准确率: </span>
                        <span style="font-weight:600">${(metrics.accuracy * 100).toFixed(2)}%</span>
                    </div>
                `;
            }
        }
        
        return `
            <div class="task-card" data-id="${task.id}">
                <div class="task-header">
                    <div>
                        <h4 class="task-name">${task.name}</h4>
                    </div>
                    ${statusBadge}
                </div>
                <div class="task-info">
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        模型: ${task.model_name || '-'}
                    </p>
                    <p style="font-size:0.875rem;color:var(--text-secondary)">
                        基准: ${task.benchmark || 'MMLU'}
                    </p>
                </div>
                ${metricsHtml}
                <div class="task-actions" style="margin-top:var(--spacing-sm)">
                    ${actionsHtml}
                </div>
            </div>
        `;
    },
    
    // =========== 用户卡片 ===========
    userCard(user) {
        return `
            <tr>
                <td>${user.id}</td>
                <td>${user.username}</td>
                <td>${user.email || '-'}</td>
                <td>${user.is_admin ? '<span style="color:var(--success)">是</span>' : '否'}</td>
                <td>${Utils.format.date(user.created_at)}</td>
                <td>
                    <button class="btn btn-sm btn-danger" onclick="Pages.admin.deleteUser(${user.id})" ${user.is_admin ? 'disabled' : ''}>
                        <span class="material-icons">delete</span>
                    </button>
                </td>
            </tr>
        `;
    },
    
    // =========== 空状态 ===========
    emptyState(icon, message, actionHtml = '') {
        return `
            <div class="empty-state">
                <span class="material-icons">${icon}</span>
                <p>${message}</p>
                ${actionHtml}
            </div>
        `;
    },
    
    // =========== 加载状态 ===========
    loading(message = '加载中') {
        return `<div class="loading">${message}</div>`;
    },
    
    // =========== 卡片容器 ===========
    card(title, content, icon = '', headerActions = '') {
        return `
            <div class="card">
                <div class="card-header">
                    <h3>
                        ${icon ? `<span class="material-icons">${icon}</span>` : ''}
                        ${title}
                    </h3>
                    ${headerActions}
                </div>
                <div class="card-body">
                    ${content}
                </div>
            </div>
        `;
    },
    
    // =========== 工具栏 ===========
    toolbar(leftContent = '', rightContent = '') {
        return `
            <div class="toolbar">
                <div class="toolbar-left">${leftContent}</div>
                <div class="toolbar-right">${rightContent}</div>
            </div>
        `;
    },
    
    // =========== 日志条目 ===========
    logEntry(log) {
        const levelClass = log.level ? log.level.toLowerCase() : 'info';
        // 后端返回 timestamp 和 content，前端兼容两种格式
        const time = log.time || log.timestamp || '';
        const message = log.message || log.content || '';
        const formattedTime = time ? Utils.format.date(time, 'HH:mm:ss') : '';
        return `
            <div class="log-entry ${levelClass}">
                <span class="log-time">${formattedTime}</span>
                ${log.level ? `<span class="log-level ${levelClass}">${log.level}</span>` : ''}
                <span class="log-message">${message}</span>
            </div>
        `;
    },
    
    // =========== 信息项 ===========
    infoItem(label, value) {
        return `
            <div class="info-item">
                <label>${label}</label>
                <span>${value || '-'}</span>
            </div>
        `;
    },
    
    // =========== GPU 卡片 ===========
    gpuCard(gpu, index) {
        const usedPercent = gpu.memory_total > 0 ? (gpu.memory_used / gpu.memory_total * 100).toFixed(1) : 0;
        // 后端返回的 memory_used 和 memory_total 是 GB 单位，需要转换成字节
        const usedBytes = gpu.memory_used * 1024 * 1024 * 1024;
        const totalBytes = gpu.memory_total * 1024 * 1024 * 1024;
        return `
            <div class="gpu-card-item">
                <div class="gpu-name">GPU ${index}: ${gpu.name}</div>
                <div class="gpu-memory">
                    显存: ${Utils.format.fileSize(usedBytes)} / ${Utils.format.fileSize(totalBytes)} (${usedPercent}%)
                </div>
                <div class="progress-bar" style="margin-top:var(--spacing-xs)">
                    <div class="progress-fill" style="width: ${usedPercent}%"></div>
                </div>
            </div>
        `;
    },
    
    // =========== 聊天消息 ===========
    chatMessage(message, role = 'user') {
        const icon = role === 'user' ? 'person' : 'smart_toy';
        return `
            <div class="message ${role}">
                <div class="message-avatar">
                    <span class="material-icons">${icon}</span>
                </div>
                <div class="message-content">${this.formatMarkdown(message)}</div>
            </div>
        `;
    },
    
    // =========== 简单 Markdown 格式化 ===========
    formatMarkdown(text) {
        if (!text) return '';
        
        // 转义 HTML
        let html = text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
        
        // 代码块
        html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
        
        // 行内代码
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // 粗体
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        // 斜体
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        
        // 换行
        html = html.replace(/\n/g, '<br>');
        
        return html;
    }
};

// 导出
window.Components = Components;
