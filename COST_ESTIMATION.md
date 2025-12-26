# Estimation du coût de production - FUTO French LLM

Analyse détaillée des coûts pour produire le modèle de langue français pour FUTO Keyboard.

---

## 📊 Résumé exécutif

| Scénario | Coût total | Temps total | Coût/heure |
|----------|------------|-------------|------------|
| **Location GPU Cloud (B200)** | **80-120€** | 20h | 4-6€/h |
| **Location GPU Cloud optimisé (A100)** | **40-60€** | 15h | 3-4€/h |
| **Achat GPU local (RTX 4090)** | **1700€ + 6€ élec** | 30h | ~57€/h* |

*Amorti sur 1 usage. Se rentabilise après 30 usages similaires.

---

## 💰 Option 1: Location GPU Cloud (Recommandé pour démarrer)

### Scénario A: GPU B200 (original)

#### Providers et tarifs

| Provider | GPU | VRAM | Prix/h | Disponibilité |
|----------|-----|------|--------|---------------|
| **RunPod** | B200 | 192 GB | $6.00/h | Bonne |
| **Lambda Labs** | B200 | 192 GB | $5.50/h | Moyenne |
| **Paperspace** | B200 | 192 GB | $5.00/h | Faible |

**Prix moyen**: ~$5.50/h (~5.20€/h au taux actuel)

#### Détail des coûts

```
Pipeline AVANT optimisation:
─────────────────────────────────────
Préentraînement:        12h × 5.20€ = 62.40€
Finetune Phase 1:        6h × 5.20€ = 31.20€
Finetune Phase 2:        8h × 5.20€ = 41.60€
Évaluation:              1h × 5.20€ =  5.20€
─────────────────────────────────────
Total:                   27h        = 140.40€

Pipeline APRÈS optimisation (HF Skills):
─────────────────────────────────────
Préentraînement:         6h × 5.20€ = 31.20€
Finetune SFT Phase 1:    3h × 5.20€ = 15.60€
Finetune SFT Phase 2:    4h × 5.20€ = 20.80€
DPO (optionnel):         2h × 5.20€ = 10.40€
Évaluation:            0.5h × 5.20€ =  2.60€
─────────────────────────────────────
Total sans DPO:       13.5h        = 70.20€
Total avec DPO:       15.5h        = 80.60€
```

**💡 Économie avec optimisations: 60€ (-43%)**

### Scénario B: GPU A100 80GB (meilleur rapport qualité/prix)

| Provider | GPU | VRAM | Prix/h |
|----------|-----|------|--------|
| **RunPod** | A100 80GB | 80 GB | $1.89/h |
| **Lambda Labs** | A100 80GB | 80 GB | $1.99/h |
| **Vast.ai** | A100 80GB | 80 GB | $1.20/h |

**Prix moyen**: ~$1.60/h (~1.50€/h)

#### Détail des coûts

```
Pipeline APRÈS optimisation sur A100:
─────────────────────────────────────
Préentraînement:         8h × 1.50€ = 12.00€
Finetune SFT Phase 1:    4h × 1.50€ =  6.00€
Finetune SFT Phase 2:    5h × 1.50€ =  7.50€
DPO (optionnel):         3h × 1.50€ =  4.50€
Évaluation:              1h × 1.50€ =  1.50€
─────────────────────────────────────
Total sans DPO:         18h        = 27.00€
Total avec DPO:         21h        = 31.50€
```

**🏆 MEILLEURE OPTION CLOUD: A100 80GB**

### Scénario C: GPU moins chers (budget serré)

| Provider | GPU | VRAM | Prix/h | Temps estimé | Coût total |
|----------|-----|------|--------|--------------|------------|
| **RunPod** | RTX 4090 | 24 GB | $0.69/h | 30h | **20.70€** |
| **Vast.ai** | RTX 4090 | 24 GB | $0.44/h | 30h | **13.20€** |
| **RunPod** | RTX 3090 | 24 GB | $0.49/h | 35h | **17.15€** |

**💸 Option ultra-économique: RTX 4090 sur Vast.ai = 13€**

---

## 🏠 Option 2: Achat GPU local

### Investissement initial

```
Configuration complète RTX 4090:
─────────────────────────────────────
GPU RTX 4090 24GB:              1700€
CPU Ryzen 7 7700X:               350€
RAM 64GB DDR5:                   200€
Carte mère B650:                 200€
SSD NVMe 2TB:                    150€
Alimentation 850W Gold:          130€
Boîtier ATX:                      80€
─────────────────────────────────────
Total matériel:                 2810€
```

### Coûts opérationnels par entraînement

```
Consommation électrique:
─────────────────────────────────────
RTX 4090: 450W en charge
Système complet: ~550W

Durée totale: 30h
Consommation: 30h × 0.55 kW = 16.5 kWh
Coût élec (0.20€/kWh): 16.5 × 0.20 = 3.30€

Usure composants (amortissement):
─────────────────────────────────────
Durée de vie GPU: ~30,000h
Utilisation: 30h
Amortissement: (1700€ / 30000h) × 30h = 1.70€

Coût total par entraînement:
─────────────────────────────────────
Électricité:                     3.30€
Amortissement:                   1.70€
─────────────────────────────────────
Total:                           5.00€
```

### Seuil de rentabilité

```
Coût initial: 2810€
Coût par entraînement local: 5€
Coût location A100 (équivalent): 27€

Économie par usage: 27€ - 5€ = 22€
Nombre d'usages pour amortir: 2810€ / 22€ = 128 usages

OU

Comparé à B200 optimisé (70€):
Économie par usage: 70€ - 5€ = 65€
Nombre d'usages pour amortir: 2810€ / 65€ = 43 usages

OU

Comparé à RTX 4090 cloud (21€):
Économie par usage: 21€ - 5€ = 16€
Nombre d'usages pour amortir: 2810€ / 16€ = 176 usages
```

**💡 Conclusion**: Rentable après ~43-130 entraînements selon scénario

---

## 📦 Option 3: Coûts additionnels (tous scénarios)

### Données et stockage

```
Téléchargement corpus:
─────────────────────────────────────
Wikipedia FR (2GB):                GRATUIT
OSCAR FR (10GB subset):            GRATUIT
mC4 FR (5GB subset):               GRATUIT

Bande passante:
Si download 17GB sur cloud:        ~0.50€

Stockage cloud (si nécessaire):
─────────────────────────────────────
100 GB pendant 1 mois:             ~2.00€
(RunPod, Lambda Labs)

Stockage local:
Déjà inclus dans config (2TB SSD)   0.00€
```

### Outils et services

```
HuggingFace Hub (hosting modèle):
─────────────────────────────────────
Free tier (illimité):              GRATUIT

Weights & Biases (monitoring):
─────────────────────────────────────
Free tier (100 GB):                GRATUIT
Pro (illimité): 50$/mois           50.00€ (opt.)

GitHub (code):
─────────────────────────────────────
Public repo:                       GRATUIT
```

---

## 📊 Comparaison totale des scénarios

### Pour UN entraînement complet

| Scénario | Setup | Entraînement | Stockage | Total | Temps |
|----------|-------|--------------|----------|-------|-------|
| **B200 Cloud** | 0€ | 70€ | 2€ | **72€** | 15h |
| **A100 Cloud** | 0€ | 27€ | 2€ | **29€** | 18h |
| **RTX 4090 Cloud (Vast.ai)** | 0€ | 13€ | 2€ | **15€** | 30h |
| **RTX 4090 Local** | 2810€ | 5€ | 0€ | **2815€** | 30h |

### Pour 10 entraînements (itérations, versions)

| Scénario | Setup | Entraînement | Total | Coût/modèle |
|----------|-------|--------------|-------|-------------|
| **B200 Cloud** | 0€ | 700€ | **700€** | 70€ |
| **A100 Cloud** | 0€ | 270€ | **270€** | 27€ |
| **RTX 4090 Cloud** | 0€ | 130€ | **130€** | 13€ |
| **RTX 4090 Local** | 2810€ | 50€ | **2860€** | 286€ |

### Pour 50 entraînements (usage professionnel)

| Scénario | Setup | Entraînement | Total | Coût/modèle |
|----------|-------|--------------|-------|-------------|
| **B200 Cloud** | 0€ | 3500€ | **3500€** | 70€ |
| **A100 Cloud** | 0€ | 1350€ | **1350€** | 27€ |
| **RTX 4090 Cloud** | 0€ | 650€ | **650€** | 13€ |
| **RTX 4090 Local** | 2810€ | 250€ | **3060€** | **61€** ⚡ |

**💡 Seuil de rentabilité GPU local: ~45 entraînements**

---

## 🎯 Recommandations par cas d'usage

### Cas 1: Projet unique / POC
**Recommandation**: RTX 4090 sur Vast.ai
- **Coût**: ~15€
- **Temps**: 30h
- **Avantage**: Minimum d'investissement

### Cas 2: 2-5 versions/itérations
**Recommandation**: A100 80GB sur RunPod/Lambda
- **Coût**: ~27-135€ (1-5 runs)
- **Temps**: 18h par run
- **Avantage**: Bon équilibre vitesse/coût

### Cas 3: Développement actif (10-20 versions)
**Recommandation**: A100 location OU achat RTX 4090
- **Coût location**: ~270-540€
- **Coût achat**: 2810€ (+ 50-100€ élec)
- **Seuil rentabilité**: ~11 runs
- **Avantage achat**: Flexibilité totale

### Cas 4: Production / Entreprise (50+ modèles)
**Recommandation**: Achat RTX 4090 OU serveur dédié
- **Coût**: 2810€ initial + 250€ élec
- **Économie vs cloud**: ~400€ sur 50 runs
- **Avantage**: Infrastructure permanente

### Cas 5: Startup ML / Lab de recherche
**Recommandation**: Multi-GPU local (2-4× RTX 4090)
- **Coût**: 5000-8000€
- **Avantage**: Parallélisation, expérimentations multiples

---

## 💡 Optimisations pour réduire les coûts

### 1. Utiliser les optimisations HF Skills
- **Économie**: 40-50% de temps GPU
- **Application**: Immédiate
- **Gain**: ~30-40€ par run

### 2. Spot instances (cloud)
```
Prix spot vs on-demand:
─────────────────────────────────────
A100 on-demand:    $1.60/h
A100 spot:         $0.80/h (-50%)
B200 on-demand:    $5.50/h
B200 spot:         $2.75/h (-50%)

Risque: Interruption possible
Mitigation: Checkpoints fréquents
```

**Économie potentielle**: **50% du coût GPU**

### 3. Préemptible/Interruptible instances
- Google Cloud: -80% du prix
- Azure Low Priority: -60-80%
- Risque d'interruption mais checkpoints auto

### 4. Réduire la taille du modèle
```
Modèle 36M → 20M paramètres:
─────────────────────────────────────
Temps entraînement: -40%
VRAM nécessaire: -35%
Qualité: -5-10% (acceptable pour MVP)

Économie: ~25€ par run
```

### 5. Utiliser des datasets plus petits (phase test)
```
Phase développement:
─────────────────────────────────────
Dataset complet: 5 GB → 18h
Dataset test: 500 MB → 2h (-90%)

Économie phase test: 24€ par itération
```

### 6. Cache et réutilisation
```
Tokenization cache:
─────────────────────────────────────
Première fois: 2h
Runs suivants: 5min

Économie: ~3€ par run (sauf le premier)
```

---

## 📈 Projection financière (12 mois)

### Scénario A: Projet hobby (1 modèle/mois)

```
Mois 1-12 (12 modèles):
─────────────────────────────────────
Option cloud (A100):  12 × 27€ = 324€
Option local (4090):  2810€ + 60€ = 2870€

Verdict: Cloud moins cher
```

### Scénario B: Startup (1 modèle/semaine)

```
Mois 1-12 (52 modèles):
─────────────────────────────────────
Option cloud (A100):  52 × 27€ = 1404€
Option local (4090):  2810€ + 260€ = 3070€

Verdict: Cloud moins cher (mais proche)
```

### Scénario C: Entreprise ML (10 modèles/semaine)

```
Mois 1-12 (520 modèles):
─────────────────────────────────────
Option cloud (A100):  520 × 27€ = 14,040€
Option local (4× 4090): 11,000€ + 2,600€ = 13,600€

Verdict: Local nettement moins cher
Économie: 440€/an
```

---

## 🎯 Verdict final

### Pour VOTRE projet FUTO French LLM

#### Première version (POC)
**Recommandation**: **A100 80GB sur RunPod**
```
Coût: 27€
Temps: 18h
Flexibilité: Maximale
Risque: Minimal
```

#### Si succès → itérations (5-10 versions)
**Recommandation**: **Continuer A100 OU acheter RTX 4090**
```
Coût total 10 runs:
- Cloud: 270€
- Local: 2810€ + 50€ = 2860€

Décision: Cloud pour <50 runs
```

#### Production / Long terme
**Recommandation**: **Achat RTX 4090**
```
ROI: 43 entraînements
Timeline: 6-12 mois selon usage
Avantages:
- Pas de limite de temps
- Expérimentations illimitées
- Autres projets ML
```

---

## 💰 Estimation finale pour VOTRE cas

### Coût de production d'un modèle FUTO French LLM

| Composante | Coût |
|------------|------|
| **GPU cloud (A100 18h)** | 27.00€ |
| **Stockage temporaire** | 2.00€ |
| **Bande passante** | 0.50€ |
| **Temps développeur** | 0€ (vous) |
| **Outils** | 0€ (gratuit) |
| **─────────────────** | **─────** |
| **TOTAL PRODUCTION** | **~30€** |

### Avec optimisations HF Skills
```
Temps réduit: 18h → 15h
Coût GPU: 15h × 1.50€ = 22.50€

TOTAL OPTIMISÉ: ~25€
```

### Budget recommandé (avec marge)
```
1 modèle (POC):              30-50€
3-5 itérations (MVP):        100-150€
10 versions (production):    250-350€
```

**🎯 Budget conseillé pour démarrer: 100-150€**

Cela vous permet de:
- ✅ Tester 3-5 configurations
- ✅ Itérer sur les hyperparamètres
- ✅ Valider la qualité
- ✅ Avoir une version production

---

**Questions ?** Quel scénario correspond le mieux à votre projet ?
