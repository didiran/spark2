"""
CI-версия тестов - проверяет только код, не требует запущенного API сервера
"""
import sys
import os

def test_imports():
    """Проверка что все модули импортируются"""
    print("\n[1] Checking imports...")
    try:
        import requests
        print("  ✅ requests module available")
        return True
    except ImportError as e:
        print(f"  ❌ Failed: {e}")
        return False

def test_syntax():
    """Проверка синтаксиса test_api.py"""
    print("\n[2] Checking test_api.py syntax...")
    try:
        with open("test_api.py", "r") as f:
            code = f.read()
        compile(code, "test_api.py", "exec")
        print("  ✅ test_api.py syntax is valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ Syntax error: {e}")
        return False

def test_web_app_imports():
    """Проверка импортов в web_app"""
    print("\n[3] Checking web_app imports...")
    try:
        # Проверяем наличие основных файлов
        required_files = [
            "web_app/app/main.py",
            "web_app/app/auth.py",
            "web_app/app/database.py",
            "web_app/app/models.py",
            "web_app/app/routers/auth.py",
            "web_app/app/routers/transactions.py",
        ]
        for file in required_files:
            if os.path.exists(file):
                print(f"  ✅ {file} exists")
            else:
                print(f"  ❌ {file} missing")
                return False
        return True
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return False

def main():
    print("=" * 50)
    print("CI Tests (No API Required)")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_syntax())
    results.append(test_web_app_imports())
    
    passed = sum(results)
    failed = len(results) - passed
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)