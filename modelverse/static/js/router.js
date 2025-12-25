/**
 * AI-Tripod 路由器
 * 基于 Hash 的 SPA 路由
 */

class Router {
    constructor() {
        this.routes = {};
        this.currentRoute = null;
        this.beforeEach = null;
        
        // 监听 hash 变化
        window.addEventListener('hashchange', () => this.handleRoute());
        window.addEventListener('load', () => this.handleRoute());
    }
    
    // 注册路由
    register(path, handler, options = {}) {
        this.routes[path] = { handler, ...options };
    }
    
    // 注册多个路由
    registerAll(routes) {
        Object.entries(routes).forEach(([path, config]) => {
            if (typeof config === 'function') {
                this.register(path, config);
            } else {
                this.register(path, config.handler, config);
            }
        });
    }
    
    // 设置路由守卫
    setBeforeEach(guard) {
        this.beforeEach = guard;
    }
    
    // 导航到指定路由
    navigate(path) {
        window.location.hash = path;
    }
    
    // 获取当前路由参数
    getParams() {
        const hash = window.location.hash.slice(1) || '/';
        const [path, queryString] = hash.split('?');
        const params = Utils.parseQuery(queryString || '');
        
        // 解析路径参数
        for (const [routePath, config] of Object.entries(this.routes)) {
            const match = this.matchRoute(routePath, path);
            if (match) {
                return { ...params, ...match.params };
            }
        }
        
        return params;
    }
    
    // 路由匹配
    matchRoute(routePath, actualPath) {
        const routeParts = routePath.split('/');
        const actualParts = actualPath.split('/');
        
        if (routeParts.length !== actualParts.length) {
            return null;
        }
        
        const params = {};
        
        for (let i = 0; i < routeParts.length; i++) {
            if (routeParts[i].startsWith(':')) {
                // 路径参数
                params[routeParts[i].slice(1)] = actualParts[i];
            } else if (routeParts[i] !== actualParts[i]) {
                return null;
            }
        }
        
        return { params };
    }
    
    // 处理路由
    async handleRoute() {
        const hash = window.location.hash.slice(1) || '/';
        const [path] = hash.split('?');
        
        let matchedRoute = null;
        let matchedPath = null;
        let params = {};
        
        // 查找匹配的路由
        for (const [routePath, config] of Object.entries(this.routes)) {
            const match = this.matchRoute(routePath, path);
            if (match) {
                matchedRoute = config;
                matchedPath = routePath;
                params = match.params;
                break;
            }
        }
        
        // 默认路由
        if (!matchedRoute) {
            matchedRoute = this.routes['/'] || this.routes['/login'];
            matchedPath = '/';
        }
        
        // 路由守卫
        if (this.beforeEach) {
            const result = await this.beforeEach({
                path,
                matchedPath,
                params,
                route: matchedRoute
            });
            
            if (result === false) {
                return;
            }
            
            if (typeof result === 'string') {
                this.navigate(result);
                return;
            }
        }
        
        // 更新当前路由
        this.currentRoute = { path, matchedPath, params, route: matchedRoute };
        
        // 执行路由处理器
        if (matchedRoute && matchedRoute.handler) {
            try {
                await matchedRoute.handler(params);
            } catch (error) {
                console.error('Route handler error:', error);
                Utils.toast.error('页面加载失败');
            }
        }
        
        // 更新导航状态
        this.updateNavigation(path);
    }
    
    // 更新导航高亮
    updateNavigation(currentPath) {
        const navLinks = document.querySelectorAll('.nav-link');
        navLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href) {
                const linkPath = href.replace('#', '');
                link.classList.toggle('active', currentPath.startsWith(linkPath));
            }
        });
    }
    
    // 刷新当前路由
    refresh() {
        this.handleRoute();
    }
}

// 创建全局路由器实例
window.router = new Router();
