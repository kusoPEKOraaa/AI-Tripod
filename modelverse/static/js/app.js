/**
 * AI-Tripod 应用主入口
 * 初始化应用和路由
 */

// 全局状态
const App = {
    user: null,
    
    // 初始化应用
    async init() {
        console.log('AI-Tripod 初始化...');
        
        // 初始化工具
        Utils.toast.init();
        Utils.modal.init();
        
        // 配置路由
        this.setupRouter();
        
        // 配置用户菜单
        this.setupUserMenu();
        
        // 检查登录状态
        if (API.isLoggedIn()) {
            try {
                this.user = await API.auth.getCurrentUser();
                this.updateUserInfo();
            } catch (error) {
                // Token 可能已过期
                API.clearToken();
            }
        }
        
        console.log('AI-Tripod 初始化完成');
    },
    
    // 配置路由
    setupRouter() {
        // 路由守卫
        router.setBeforeEach(async (to) => {
            const publicRoutes = ['/login', '/register'];
            const isPublic = publicRoutes.includes(to.path);
            const isLoggedIn = API.isLoggedIn();
            
            // 未登录访问需要认证的页面
            if (!isPublic && !isLoggedIn) {
                return '/login';
            }
            
            // 已登录访问登录/注册页面
            if (isPublic && isLoggedIn) {
                return '/home';
            }
            
            // 管理员页面权限检查
            if (to.path === '/admin') {
                if (!this.user) {
                    try {
                        this.user = await API.auth.getCurrentUser();
                    } catch (error) {
                        return '/login';
                    }
                }
                if (!this.user.is_admin) {
                    Utils.toast.error('无权访问此页面');
                    return '/home';
                }
            }
            
            return true;
        });
        
        // 注册路由
        router.registerAll({
            '/': { handler: () => Pages.home.render() },
            '/home': { handler: () => Pages.home.render() },
            '/login': { handler: () => Pages.auth.renderLogin() },
            '/register': { handler: () => Pages.auth.renderRegister() },
            '/dashboard': { handler: () => Pages.dashboard.render() },
            '/resources': { handler: () => Pages.resources.render() },
            '/training': { handler: () => Pages.training.render() },
            '/training/:id': { handler: (params) => Pages.training.renderDetail(params) },
            '/inference': { handler: () => Pages.inference.render() },
            '/chat/:id': { handler: (params) => Pages.chat.render(params) },
            '/evaluation': { handler: () => Pages.evaluation.render() },
            '/evaluation/:id': { handler: (params) => Pages.evaluation.renderDetail(params) },
            '/profile': { handler: () => Pages.profile.render() },
            '/admin': { handler: () => Pages.admin.render() }
        });

        // 首次渲染：注册完路由后立刻处理当前 hash
        router.handleRoute();
    },
    
    // 配置用户菜单
    setupUserMenu() {
        const userBtn = document.getElementById('user-btn');
        const userMenu = document.getElementById('user-menu');
        
        if (userBtn && userMenu) {
            // 点击用户按钮切换菜单
            userBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                userMenu.classList.toggle('show');
            });
            
            // 点击其他地方关闭菜单
            document.addEventListener('click', () => {
                userMenu.classList.remove('show');
            });
        }
        
        // 登出按钮
        const logoutBtn = document.getElementById('logout-btn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.logout();
            });
        }
    },
    
    // 更新用户信息
    updateUserInfo() {
        const usernameEl = document.getElementById('username');
        if (usernameEl && this.user) {
            usernameEl.textContent = this.user.username;
        }
        
        // 管理员标记
        if (this.user && this.user.is_admin) {
            document.body.classList.add('is-admin');
        } else {
            document.body.classList.remove('is-admin');
        }
    },
    
    // 登出
    logout() {
        API.auth.logout();
        this.user = null;
        document.body.classList.remove('is-admin');
        Utils.toast.success('已退出登录');
    }
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// 导出
window.App = App;
