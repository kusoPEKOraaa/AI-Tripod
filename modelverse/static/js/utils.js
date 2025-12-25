/**
 * AI-Tripod 工具函数
 * Toast 通知、Modal 对话框、格式化等
 */

const Utils = {
    // =========== Toast 通知 ===========
    toast: {
        container: null,
        
        // 初始化 toast 容器
        init() {
            this.container = document.getElementById('toast-container');
            if (!this.container) {
                this.container = document.createElement('div');
                this.container.id = 'toast-container';
                this.container.className = 'toast-container';
                document.body.appendChild(this.container);
            }
        },
        
        // 显示 toast
        show(message, type = 'info', duration = 3000) {
            if (!this.container) this.init();
            
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            
            const icons = {
                success: 'check_circle',
                error: 'error',
                warning: 'warning',
                info: 'info'
            };
            
            toast.innerHTML = `
                <span class="toast-icon material-icons">${icons[type]}</span>
                <div class="toast-content">
                    <div class="toast-message">${message}</div>
                </div>
                <button class="toast-close" onclick="this.parentElement.remove()">
                    <span class="material-icons">close</span>
                </button>
            `;
            
            this.container.appendChild(toast);
            
            // 自动移除
            if (duration > 0) {
                setTimeout(() => {
                    toast.style.animation = 'slideOut 0.3s ease forwards';
                    setTimeout(() => toast.remove(), 300);
                }, duration);
            }
            
            return toast;
        },
        
        success(message, duration) {
            return this.show(message, 'success', duration);
        },
        
        error(message, duration = 5000) {
            return this.show(message, 'error', duration);
        },
        
        warning(message, duration) {
            return this.show(message, 'warning', duration);
        },
        
        info(message, duration) {
            return this.show(message, 'info', duration);
        }
    },
    
    // =========== Modal 对话框 ===========
    modal: {
        overlay: null,
        
        // 初始化 modal 容器
        init() {
            this.overlay = document.getElementById('modal-overlay');
            if (!this.overlay) {
                this.overlay = document.createElement('div');
                this.overlay.id = 'modal-overlay';
                this.overlay.className = 'modal-overlay';
                this.overlay.innerHTML = `
                    <div class="modal">
                        <div class="modal-header">
                            <h3 class="modal-title"></h3>
                            <button class="modal-close" onclick="Utils.modal.close()">
                                <span class="material-icons">close</span>
                            </button>
                        </div>
                        <div class="modal-body"></div>
                        <div class="modal-footer"></div>
                    </div>
                `;
                document.body.appendChild(this.overlay);
                
                // 点击遮罩关闭
                this.overlay.addEventListener('click', (e) => {
                    if (e.target === this.overlay) {
                        this.close();
                    }
                });
            }
        },
        
        // 打开 modal
        open(options = {}) {
            if (!this.overlay) this.init();
            
            const {
                title = '',
                content = '',
                footer = '',
                onClose = null,
                size = 'medium'
            } = options;
            
            const modal = this.overlay.querySelector('.modal');
            modal.querySelector('.modal-title').textContent = title;
            modal.querySelector('.modal-body').innerHTML = content;
            modal.querySelector('.modal-footer').innerHTML = footer;
            
            // 设置大小
            modal.style.maxWidth = size === 'large' ? '800px' : size === 'small' ? '400px' : '500px';
            
            this.overlay.classList.add('show');
            this.onClose = onClose;
            
            // 禁止背景滚动
            document.body.style.overflow = 'hidden';
        },
        
        // 关闭 modal
        close() {
            if (this.overlay) {
                this.overlay.classList.remove('show');
                document.body.style.overflow = '';
                if (this.onClose) {
                    this.onClose();
                    this.onClose = null;
                }
            }
        },
        
        // 确认对话框
        confirm(message, onConfirm, onCancel) {
            this.open({
                title: '确认',
                content: `<p>${message}</p>`,
                footer: `
                    <button class="btn btn-secondary" onclick="Utils.modal.close(); ${onCancel ? onCancel.toString() + '()' : ''}">取消</button>
                    <button class="btn btn-danger" id="modal-confirm-btn">确定</button>
                `,
                size: 'small'
            });
            
            document.getElementById('modal-confirm-btn').addEventListener('click', () => {
                this.close();
                if (onConfirm) onConfirm();
            });
        },
        
        // 表单对话框
        form(options = {}) {
            const { title, fields = [], onSubmit, submitText = '提交' } = options;
            
            let formHtml = '<form id="modal-form">';
            fields.forEach(field => {
                formHtml += `
                    <div class="form-group">
                        <label>${field.label}${field.required ? ' <span style="color:var(--error)">*</span>' : ''}</label>
                `;
                
                if (field.type === 'select') {
                    formHtml += `<select class="form-control form-select" name="${field.name}" ${field.required ? 'required' : ''}>`;
                    field.options.forEach(opt => {
                        formHtml += `<option value="${opt.value}" ${opt.value === field.value ? 'selected' : ''}>${opt.label}</option>`;
                    });
                    formHtml += '</select>';
                } else if (field.type === 'textarea') {
                    formHtml += `<textarea class="form-control" name="${field.name}" ${field.required ? 'required' : ''} placeholder="${field.placeholder || ''}">${field.value || ''}</textarea>`;
                } else {
                    formHtml += `<input type="${field.type || 'text'}" class="form-control" name="${field.name}" value="${field.value || ''}" ${field.required ? 'required' : ''} placeholder="${field.placeholder || ''}">`;
                }
                
                formHtml += '</div>';
            });
            formHtml += '</form>';
            
            this.open({
                title,
                content: formHtml,
                footer: `
                    <button class="btn btn-secondary" onclick="Utils.modal.close()">取消</button>
                    <button class="btn btn-primary" id="modal-submit-btn">${submitText}</button>
                `,
                size: options.size || 'medium'
            });
            
            document.getElementById('modal-submit-btn').addEventListener('click', async () => {
                const form = document.getElementById('modal-form');
                if (!form.checkValidity()) {
                    form.reportValidity();
                    return;
                }
                
                const formData = new FormData(form);
                const data = Object.fromEntries(formData.entries());
                
                try {
                    await onSubmit(data);
                    this.close();
                } catch (error) {
                    Utils.toast.error(error.message);
                }
            });
        }
    },
    
    // =========== 格式化 ===========
    format: {
        // 日期格式化
        date(dateStr, format = 'YYYY-MM-DD HH:mm:ss') {
            if (!dateStr) return '-';
            // 处理后端返回的时间格式（可能没有时区信息）
            let date;
            if (typeof dateStr === 'string' && !dateStr.includes('T') && !dateStr.includes('Z')) {
                // 格式如 "2025-12-25 10:05:01"，假定为UTC，转换为本地时间
                date = new Date(dateStr.replace(' ', 'T') + 'Z');
            } else {
                date = new Date(dateStr);
            }
            
            // 检查日期是否有效
            if (isNaN(date.getTime())) return '-';
            
            const year = date.getFullYear();
            const month = String(date.getMonth() + 1).padStart(2, '0');
            const day = String(date.getDate()).padStart(2, '0');
            const hours = String(date.getHours()).padStart(2, '0');
            const minutes = String(date.getMinutes()).padStart(2, '0');
            const seconds = String(date.getSeconds()).padStart(2, '0');
            
            return format
                .replace('YYYY', year)
                .replace('MM', month)
                .replace('DD', day)
                .replace('HH', hours)
                .replace('mm', minutes)
                .replace('ss', seconds);
        },
        
        // 相对时间
        relativeTime(dateStr) {
            if (!dateStr) return '-';
            // 处理后端返回的时间格式
            let date;
            if (typeof dateStr === 'string' && !dateStr.includes('T') && !dateStr.includes('Z')) {
                date = new Date(dateStr.replace(' ', 'T') + 'Z');
            } else {
                date = new Date(dateStr);
            }
            
            if (isNaN(date.getTime())) return '-';
            
            const now = new Date();
            const diff = now - date;
            
            const seconds = Math.floor(diff / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);
            const days = Math.floor(hours / 24);
            
            if (days > 7) {
                return this.date(dateStr, 'YYYY-MM-DD');
            } else if (days > 0) {
                return `${days}天前`;
            } else if (hours > 0) {
                return `${hours}小时前`;
            } else if (minutes > 0) {
                return `${minutes}分钟前`;
            } else {
                return '刚刚';
            }
        },
        
        // 文件大小
        fileSize(bytes) {
            if (!bytes) return '0 B';
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            let index = 0;
            let size = bytes;
            
            while (size >= 1024 && index < units.length - 1) {
                size /= 1024;
                index++;
            }
            
            return `${size.toFixed(2)} ${units[index]}`;
        },
        
        // 时长格式化
        duration(seconds) {
            if (!seconds) return '0秒';
            
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);
            
            const parts = [];
            if (hours > 0) parts.push(`${hours}小时`);
            if (minutes > 0) parts.push(`${minutes}分钟`);
            if (secs > 0) parts.push(`${secs}秒`);
            
            return parts.join('') || '0秒';
        },
        
        // 百分比
        percent(value, decimals = 1) {
            if (value === null || value === undefined) return '-';
            return `${(value * 100).toFixed(decimals)}%`;
        },
        
        // 数字缩写
        number(num) {
            if (!num) return '0';
            if (num >= 1000000) {
                return (num / 1000000).toFixed(1) + 'M';
            } else if (num >= 1000) {
                return (num / 1000).toFixed(1) + 'K';
            }
            return num.toString();
        }
    },
    
    // =========== 状态 ===========
    status: {
        // 资源状态文本映射
        resourceStatus: {
            PENDING: { text: '待下载', class: 'status-pending' },
            DOWNLOADING: { text: '下载中', class: 'status-downloading' },
            COMPLETED: { text: '已完成', class: 'status-completed' },
            FAILED: { text: '下载失败', class: 'status-failed' },
            CANCELLED: { text: '已取消', class: 'status-stopped' }
        },
        
        // 训练状态文本映射
        trainingStatus: {
            PENDING: { text: '待启动', class: 'status-pending' },
            RUNNING: { text: '训练中', class: 'status-running' },
            COMPLETED: { text: '已完成', class: 'status-completed' },
            FAILED: { text: '训练失败', class: 'status-failed' },
            STOPPED: { text: '已停止', class: 'status-stopped' }
        },
        
        // 推理状态文本映射
        inferenceStatus: {
            STOPPED: { text: '已停止', class: 'status-stopped' },
            CREATING: { text: '创建中', class: 'status-creating' },
            RUNNING: { text: '运行中', class: 'status-running' },
            FAILED: { text: '启动失败', class: 'status-failed' }
        },
        
        // 评估状态文本映射
        evaluationStatus: {
            PENDING: { text: '待启动', class: 'status-pending' },
            RUNNING: { text: '评估中', class: 'status-running' },
            COMPLETED: { text: '已完成', class: 'status-completed' },
            FAILED: { text: '评估失败', class: 'status-failed' },
            STOPPED: { text: '已停止', class: 'status-stopped' }
        },
        
        // 获取状态徽章 HTML
        getBadge(status, type = 'resource') {
            const statusMap = this[`${type}Status`] || this.resourceStatus;
            const upperStatus = (status || '').toUpperCase();
            const info = statusMap[upperStatus] || { text: status, class: 'status-pending' };
            return `<span class="status-badge ${info.class}">${info.text}</span>`;
        }
    },
    
    // =========== DOM 辅助 ===========
    dom: {
        // 根据模板 ID 获取内容
        getTemplate(templateId) {
            const template = document.getElementById(templateId);
            return template ? template.innerHTML : '';
        },
        
        // 渲染到目标元素
        render(targetId, content) {
            const target = document.getElementById(targetId);
            if (target) {
                target.innerHTML = content;
            }
        },
        
        // 创建元素
        create(tag, attrs = {}, children = []) {
            const el = document.createElement(tag);
            Object.entries(attrs).forEach(([key, value]) => {
                if (key === 'className') {
                    el.className = value;
                } else if (key === 'style' && typeof value === 'object') {
                    Object.assign(el.style, value);
                } else if (key.startsWith('on')) {
                    el.addEventListener(key.slice(2).toLowerCase(), value);
                } else {
                    el.setAttribute(key, value);
                }
            });
            children.forEach(child => {
                if (typeof child === 'string') {
                    el.appendChild(document.createTextNode(child));
                } else if (child) {
                    el.appendChild(child);
                }
            });
            return el;
        },
        
        // 显示/隐藏元素
        show(el) {
            if (typeof el === 'string') el = document.getElementById(el);
            if (el) el.style.display = '';
        },
        
        hide(el) {
            if (typeof el === 'string') el = document.getElementById(el);
            if (el) el.style.display = 'none';
        },
        
        toggle(el, show) {
            if (show) this.show(el);
            else this.hide(el);
        }
    },
    
    // =========== 其他工具 ===========
    // 防抖
    debounce(fn, delay = 300) {
        let timer = null;
        return function(...args) {
            clearTimeout(timer);
            timer = setTimeout(() => fn.apply(this, args), delay);
        };
    },
    
    // 节流
    throttle(fn, delay = 300) {
        let last = 0;
        return function(...args) {
            const now = Date.now();
            if (now - last >= delay) {
                last = now;
                fn.apply(this, args);
            }
        };
    },
    
    // 复制到剪贴板
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.toast.success('已复制到剪贴板');
        } catch (error) {
            this.toast.error('复制失败');
        }
    },
    
    // 下载文件
    downloadFile(url, filename) {
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    },
    
    // 生成唯一 ID
    generateId() {
        return `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    },
    
    // 解析查询参数
    parseQuery(queryString) {
        const params = new URLSearchParams(queryString);
        const result = {};
        for (const [key, value] of params) {
            result[key] = value;
        }
        return result;
    }
};

// 添加 slideOut 动画
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOut {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
`;
document.head.appendChild(style);

// 导出
window.Utils = Utils;
