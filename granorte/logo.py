"""
setup.py — Preparação do sistema Granorte
Baixa a logo oficial e verifica as dependências.

Execute UMA VEZ antes de usar o sistema:
    python setup.py
"""

import sys
import urllib.request
from pathlib import Path

LOGO_URL  = "https://granortesa.ind.br/wp-content/uploads/2022/02/logo-granorte-desde72.png"
LOGO_PATH = Path("logo_granorte.png")


def baixar_logo():
    print("[1/3] Baixando logo da Granorte...")
    try:
        req = urllib.request.Request(
            LOGO_URL,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            dados = resp.read()
        LOGO_PATH.write_bytes(dados)
        print(f"      ✅  Logo salva em: {LOGO_PATH}  ({len(dados)//1024} KB)")
        return True
    except Exception as e:
        print(f"      ⚠️  Não foi possível baixar automaticamente: {e}")
        print("      👉  Acesse https://granortesa.ind.br e salve a logo manualmente")
        print(f"          como '{LOGO_PATH}' na mesma pasta do sistema.")
        return False


def verificar_dependencias():
    print("\n[2/3] Verificando dependências Python...")
    deps = {
        "cv2":         "opencv-python",
        "numpy":       "numpy",
        "ultralytics": "ultralytics",
        "easyocr":     "easyocr",
    }
    faltando = []
    for modulo, pacote in deps.items():
        try:
            __import__(modulo)
            print(f"      ✅  {pacote}")
        except ImportError:
            print(f"      ❌  {pacote}  — NÃO INSTALADO")
            faltando.append(pacote)

    if faltando:
        print(f"\n      Instale os pacotes faltando com:")
        print(f"      pip install {' '.join(faltando)}")
        return False
    return True


def criar_pastas():
    print("\n[3/3] Criando estrutura de pastas...")
    for pasta in ["registros", "registros/placas", "registros/basculantes"]:
        Path(pasta).mkdir(parents=True, exist_ok=True)
        print(f"      📁  {pasta}/")
    return True


if __name__ == "__main__":
    print("=" * 52)
    print("  SETUP — Sistema de Monitoramento Granorte S/A")
    print("=" * 52)

    ok_logo = baixar_logo()
    ok_deps = verificar_dependencias()
    ok_past = criar_pastas()

    print("\n" + "=" * 52)
    if ok_logo and ok_deps and ok_past:
        print("  ✅  Tudo pronto! Execute:")
        print("      python detector.py")
    else:
        print("  ⚠️  Resolva os itens acima e execute novamente.")
    print("=" * 52)