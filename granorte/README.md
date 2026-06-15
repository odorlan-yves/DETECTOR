# 🚛 Sistema de Monitoramento de Caminhões

Sistema de câmera de segurança para detecção de **placas** e análise do **basculante** (limpo/sujo) em tempo real.

---

## 📦 Instalação

```bash
pip install ultralytics easyocr opencv-python numpy
```

---

## 🚀 Como usar

### Sistema principal (detecção ao vivo)

```bash
# Câmeras físicas (0 = placa, 1 = basculante)
python detector.py

# Vídeos de teste
python detector.py --cam-placa video_placa.mp4 --cam-basculante video_bascul.mp4

# Câmeras IP (RTSP)
python detector.py \
  --cam-placa rtsp://192.168.1.10/stream \
  --cam-basculante rtsp://192.168.1.11/stream

# Com modelos YOLO próprios (melhora muito a detecção)
python detector.py \
  --modelo-placa placa.pt \
  --modelo-basculante basculante.pt
```

### Visualizador de histórico

```bash
# Todo o histórico
python historico.py

# Filtrar por data
python historico.py --data 2024-12-01

# Filtrar por placa
python historico.py --placa ABC1234
```

---

## 🖥️ Layout da tela principal

```
┌──────────────────────┬─────────────────┐
│  CAM 1 — PLACA       │  PLACA DETECTADA│
│  [frame ao vivo]     │  ABC1D23        │
│                      │─────────────────│
├──────────────────────│  BASCULANTE     │
│  CAM 2 — BASCULANTE  │  SUJO  ████░░  │
│  [frame ao vivo]     │─────────────────│
│  Score: 72.4/100     │  HISTÓRICO      │
│  ● SUJO              │  ...entradas... │
└──────────────────────┴─────────────────┘
```

---

## 📁 Arquivos gerados

```
registros/
├── historico.db          ← banco SQLite com tudo
├── placas/               ← fotos de cada placa detectada
│   └── ABC1D23_1733000.jpg
└── basculantes/          ← fotos do basculante quando muda status
    └── bascul_1733001.jpg
```

### Tabelas do banco

| Tabela        | O que guarda                                      |
|---------------|--------------------------------------------------|
| `placas`      | timestamp, placa, tipo (Mercosul/Antiga), confiança, foto |
| `basculantes` | timestamp, status, score de sujeira, placa associada, foto |
| `entradas`    | registro de cada caminhão: placa + estado do basculante |

---

## 🔧 Como funciona sem modelo treinado

### Detecção de placa
- Busca regiões **brancas/amarelas** no terço inferior da imagem
- Filtra por proporção típica de placa (largura/altura entre 2x e 6x)
- OCR via **EasyOCR** valida o formato brasileiro (Mercosul `ABC1D23` ou Antiga `ABC1234`)
- Votação nos últimos 12 frames para estabilizar a leitura

### Análise do basculante
Combina 4 métricas ponderadas:
| Métrica | Peso | O que detecta |
|---------|------|---------------|
| Variância de textura (Laplaciano) | 30% | Irregularidade da superfície |
| Escurecimento (canal V do HSV) | 25% | Terra e lama escuras |
| Tom marrom (Hue 8–25°) | 30% | Cor característica de terra |
| Heterogeneidade de cor | 15% | Mistura de resíduos diferentes |

**Score ≥ 55 → SUJO | 36–55 → LEVEMENTE SUJO | < 36 → LIMPO**

---

## 🎯 Para melhorar a precisão

Adicione modelos YOLO treinados especificamente:

### Modelo de placa brasileira (grátis)
1. Acesse [universe.roboflow.com](https://universe.roboflow.com)
2. Pesquise `brazilian license plate`
3. Baixe o `.pt` e use `--modelo-placa placa.pt`

### Modelo de basculante
- Treine com imagens do seu próprio ambiente
- Use o Roboflow para anotar e treinar gratuitamente
- Use `--modelo-basculante basculante.pt`

---

## ⌨️ Controles

| Tecla | Ação |
|-------|------|
| `Q` ou `ESC` | Encerrar |
| `R` (no histórico) | Atualizar dados |
| `A` / `←` (no histórico) | Página anterior |
| `D` / `→` (no histórico) | Próxima página |
