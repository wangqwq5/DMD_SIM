import ast
with open('main.py', encoding='utf-8') as f:
    src = f.read()
ast.parse(src)
print('OK')
