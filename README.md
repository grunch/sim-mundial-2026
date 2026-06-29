# Simulador Mundial 2026 — Dieciseisavos a la final

Modelo cuantitativo, reproducible y auditable para pronosticar los cruces
eliminatorios de la **Copa del Mundo FIFA 2026** (Canadá / México / EE. UU.),
desde la fase de **32 equipos (Round of 32 / dieciseisavos)** hasta el campeón.

Cada selección tiene un perfil con datos crudos (ranking FIFA, valor de plantilla,
rendimiento en grupos, palmarés) y métricas derivadas con **fórmulas explícitas**.
A partir de ahí, un motor cabeza a cabeza estima la probabilidad de que un equipo
elimine a otro, y dos simuladores resuelven el cuadro completo.

---

## 📁 Archivos del proyecto

| Archivo | Qué es |
|---|---|
| `mundial2026_r32_dataset.json` | **Fuente de verdad.** Los 32 equipos con datos crudos y métricas derivadas. |
| `mundial2026_montecarlo.json` | Salida de `simulate_bracket.py`: probabilidad de cada equipo de alcanzar cada ronda. |
| `predict.py` | Motor cabeza a cabeza: probabilidad de que A elimine a B en un duelo directo. |
| `simulate_bracket.py` | **Montecarlo agregado:** juega el torneo N veces y cuenta quién es campeón con qué frecuencia. |
| `bracket.py` | **Cuadro único predicho:** dibuja el bracket ronda por ronda; avanza el favorito y los **empates se juegan a penales por azar**, así que **cada corrida puede dar otro campeón** (`--seed` para reproducir uno). Con `--marcador [N]` añade el marcador más probable de cada partido (Poisson, % exacto). |
| `spec_mundial2026.md` | Especificación técnica completa (fórmulas, fuentes, esquema JSON). |

> ⚙️ El generador del dataset (`build_dataset.py`, citado en el spec) no está incluido;
> la fuente de verdad es el JSON, que ya trae todas las métricas calculadas.

---

## 🚀 Inicio rápido

Requisitos: **Python 3** (solo librería estándar, sin dependencias externas).

```bash
# 1) Pronóstico de un duelo directo (P de avanzar, incl. prórroga/penales)
python3 predict.py ARG CPV          # Argentina vs Cabo Verde
python3 predict.py BEL SEN 0.28     # Bélgica vs Senegal, versión 90' con banda de empate

# 2) Los 16 cruces oficiales de dieciseisavos
python3 predict.py

# 3) Montecarlo: ¿quién es campeón con más frecuencia? (N torneos)
python3 simulate_bracket.py 200000

# 4) Cuadro completo dibujado, ronda por ronda hasta la final
python3 bracket.py                  # cuadro predicho; empates a penales por azar (--seed)
python3 bracket.py --marcador       # + marcador más probable de cada cruce (2-0, 21%)
python3 bracket.py --marcador 3     # + los 3 marcadores más probables por cruce
```

---

## 🧠 Cómo funciona el modelo

### Datos crudos (4 ejes + historial)

1. **Ranking FIFA** — puntos oficiales (sistema Elo de la FIFA, denominador 600).
2. **Valor de plantilla** — Transfermarkt, en millones de EUR.
3. **Rendimiento en grupos 2026** — `points` y `goal_difference`.
4. **Palmarés mundialista** — títulos, mejor resultado histórico, participaciones.

### Métricas derivadas (todas normalizadas sobre el pool de 32)

```
norm_fifa  = minmax(puntos_oficiales_FIFA)
norm_value = minmax(log10(valor_plantilla))         # log: el valor está muy sesgado
norm_form  = minmax(form_raw),  form_raw = points + 0.4 · goal_difference
norm_pedigree = minmax(pedigree_raw)

Strength Index (0–100) = 100 · (0.40·norm_fifa + 0.20·norm_value
                                + 0.25·norm_form + 0.15·norm_pedigree)

effective_elo = puntos_FIFA + 40·z(form_raw) + 25·z(log10_value) + 20·z(pedigree_raw)
```

(`minmax` = reescala a [0,1]; `z` = puntaje estándar respecto al pool.)

### Probabilidad de un duelo (motor)

Se promedian **dos métodos independientes**:

```
Método A (Elo):  P(A) = 1 / (1 + 10^(-(eELO_A − eELO_B)/600))
Método B (SI):   P(A) = 1 / (1 + 10^(-(SI_A − SI_B)/12))
P(A avanza)  =  (P_A_elo + P_A_si) / 2          # incluye prórroga y penales
```

Detalles completos y fuentes en [`spec_mundial2026.md`](./spec_mundial2026.md).

---

## ⚔️ `simulate_bracket.py` vs `bracket.py` — leen el torneo distinto

Las dos usan **la misma probabilidad por partido**, pero responden preguntas diferentes.
Es normal que **no coincida el campeón** entre una y otra.

### `simulate_bracket.py` — "¿quién gana MÁS torneos?"

Juega el cuadro completo N veces. En cada partido tira un dado (`random() < P`),
así que **a veces el favorito pierde** (como en la vida real). Cuenta en qué
fracción de los N torneos cada equipo llega a cada ronda y es campeón.

Captura el **camino**: un equipo con sorteo fácil llega más lejos aunque no sea el
mejor en duelo directo.

### `bracket.py` — "un cuadro predicho, con los penales jugados a azar"

Resuelve **un solo cuadro** ronda por ronda. En cada cruce mira el **marcador más
probable** (modelo Poisson, ver abajo):

- Si **no** es empate → avanza el favorito (gana en los 90', determinista).
- Si **es empate** → va a **penales**, y el ganador se decide **por azar**: una
  moneda ponderada por la fuerza (`p_adv`), de modo que **el menos favorito también
  puede pasar**. Esos cruces se marcan `(pen)`.

Como los penales son azar, **cada corrida puede dar otro campeón**: si la final
queda 1-1, a veces gana Francia y a veces Argentina. Por defecto el azar es **real**
(semilla del sistema); la cabecera imprime la semilla usada para que puedas
**reproducir** ese cuadro exacto con `--seed`.

```bash
python3 bracket.py                  # cuadro NUEVO cada vez (penales al azar)
python3 bracket.py                  # ...corre de nuevo y el campeón puede cambiar
python3 bracket.py --seed 7         # reproducible: misma semilla → mismo cuadro
```

*(Para "¿quién gana MÁS torneos?" sobre miles de realizaciones como esta, usa
`simulate_bracket.py`.)*

#### Marcadores con `--marcador [N]`

Predice además el **marcador** de cada cruce con un modelo de goles **Poisson**
derivado del Elo: `λ_fav` y `λ_rival` se obtienen de la supremacía en Elo, y la
probabilidad **exacta** de cada resultado es `P(i-j) = Poisson(i; λ_fav)·Poisson(j; λ_rival)`.

- `N` = cuántos marcadores más probables mostrar por partido (por defecto 1).
- Es **analítico, no por iteraciones**: el `%` es exacto e instantáneo. Simular N
  partidos solo aproximaría ese número con ruido (con `N=3`, `2-0 (66%)` sería estadística falsa).
- El marcador más probable de un favorito aplastante **no** es 4-0: la probabilidad
  se reparte. Argentina vs Cabo Verde → más probable **2-0, pero solo 21%** (gana 98%).
- Empate como marcador más probable → se marca `→pen`: ese cruce se decide en
  **penales por azar** (ver arriba), no automáticamente al favorito.

```bash
python3 bracket.py --marcador       # [86] *Argentina vs Cabo Verde -> Argentina (97.7%) | 2-0 (21.4%)
python3 bracket.py --marcador 3     # ... | 2-0 (21.4%) · 1-0 (17.8%) · 3-0 (17.1%)
```

### Ejemplo real de la diferencia

| Pregunta | Herramienta | Campeón |
|---|---|---|
| ¿Quién levanta la copa más veces? | `simulate_bracket.py` | **Argentina** (~39%) |
| ¿Cómo queda UN cuadro (penales a azar)? | `bracket.py --seed 2026` | **Francia** |

¿Por qué? Argentina y Francia están en mitades opuestas (solo se cruzan en la final).
En la final **Francia es favorita por poco (~55%)**, así que en muchas semillas gana
Francia — pero al jugarse los empates a penales, otras semillas dan otro campeón.
Pero la mitad de Francia es brutal (España, Alemania, Países Bajos), así que **Francia
solo llega a la final el 54% de las veces**, mientras **Argentina llega el 65%** gracias
a un camino más fácil. Sobre miles de torneos, Argentina levanta la copa más seguido.

> 📌 **Ser campeón del cuadro de favoritos ≠ ser el campeón más probable.** El sorteo
> pesa tanto como la calidad.

---

## 🇸🇳 Nota sobre Senegal (grupo I)

El dataset traía una discrepancia: la tabla NBC marcaba a Senegal con `0 pts / −3 DG`,
pero el detalle de partidos registraba una **victoria 5-0 sobre Irak**. Se resolvió
adoptando el detalle de partidos:

**Senegal:** 3 PJ · 1 V · 0 E · 2 D · **3 pts** · DG **+3** (derrotas ante Francia y
Noruega + goleada a Irak).

Como Senegal tenía el mínimo del pool de `form_raw`, al corregirlo se recalcularon
`norm_form`, `SI` y `eELO` de los 32 equipos. Senegal subió del puesto 21 al 17 por
Strength Index. Queda registrado en `data_notes` del equipo y en los `caveats` del `meta`.

---

## 🔁 Reproducir todo

```bash
python3 predict.py                  # 16 pronósticos de dieciseisavos
python3 simulate_bracket.py 200000  # Montecarlo → regenera mundial2026_montecarlo.json
python3 bracket.py                  # cuadro completo dibujado (empates a penales por azar)
```

---

## ⚠️ Limitaciones

- **Torneo en curso:** las cifras y posiciones son una foto al **28–29 jun 2026**.
- El modelo es una **línea base cuantitativa**, no una verdad: no captura lesiones,
  suspensiones, descanso entre rondas, localía real, motivación ni táctica del rival.
- El `pedigree_score` es un proxy reproducible; subestima a finalistas/semifinalistas
  sin título (Países Bajos, Suecia, Portugal, Croacia).
- Valores Transfermarkt en EUR (no mezclar con listados en USD).

---

## 📐 Códigos de equipo

Se usan códigos de 3 letras (FIFA) en `predict.py` y en el dataset: `ARG`, `FRA`,
`ESP`, `BRA`, `ENG`, `GER`, `NED`, `POR`, `MAR`, `BEL`, `MEX`, `CRO`, `USA`, `SUI`,
`COL`, `JPN`, `NOR`, `CIV`, `ECU`, `AUT`, `SEN`, `SWE`, `CAN`, `ALG`, `EGY`, `AUS`,
`PAR`, `COD`, `GHA`, `BIH`, `RSA`, `CPV`.
