================================================================================
                   XAUUSD INTELLIGENT TP BOT - VERSION 3.0
                               README & TUTORIAL
================================================================================

1. INTRODUZIONE
---------------
Questo bot è un sistema di trading automatico avanzato progettato specificamente
per il simbolo XAUUSD (Oro) su piattaforma MetaTrader 5.
Utilizza un approccio ibrido che combina Machine Learning (Random Forest,
Gradient Boosting), Reinforcement Learning, analisi tecnica (ATR, RSI, Bollinger)
e rilevamento dinamico di Supporti/Resistenze per generare segnali di trading
e ottimizzare dinamicamente il livello di Take Profit (TP) e Stop Loss (SL).

Il bot è pensato per operare con un lotto fisso (configurabile) e mantiene
una sola posizione aperta alla volta per ridurre il rischio.

2. CARATTERISTICHE PRINCIPALI (FEATURES)
-----------------------------------------
✅ Connessione robusta a MT5 con auto‑recovery e health check in background.
✅ Raccolta dati multi‑timeframe (M1, M5, M15, H1).
✅ Machine Learning per segnali BUY/SELL:
   - Modello Random Forest addestrato su indicatori tecnici.
   - Soglia di confidenza configurabile (default 0.70).
✅ Sistema Ibrido di Take Profit (Hybrid TP Manager):
   - TP basato su ATR e regime di mercato.
   - TP predetto da modello ML (Gradient Boosting Regressor).
   - TP calcolato da livelli di Supporto/Resistenza dinamici.
   - TP ottimizzato con Reinforcement Learning (Q‑learning).
   - Pesi delle strategie aggiornati automaticamente in base alla performance.
✅ Analisi del regime di mercato (alta/bassa volatilità, trend, range, breakout).
✅ Reinforcement Learning Agent per aggiustamento dinamico del TP.
✅ Risk Manager avanzato con Circuit Breaker:
   - Limite di perdita giornaliera.
   - Massimo numero di perdite consecutive.
   - Massimo drawdown consentito.
✅ Monitoraggio real‑time e logging dettagliato.
✅ Salvataggio e caricamento automatico dei modelli addestrati (per riprendere
   rapidamente senza dover riaddestrare).
✅ Interfaccia a menu interattiva per setup, avvio, stop e analisi performance.

3. VERSIONI PYTHON SUPPORTATE
-----------------------------
- Python 3.7, 3.8, 3.9, 3.10, 3.11 (testato su 3.9 e 3.10)
- Architettura a 64‑bit richiesta per MetaTrader5 e TA‑Lib.

4. REQUISITI (REQUIREMENTS.TXT)
-------------------------------
Scarica i moduli attraverso file requiriments.txt

5. INSTALLAZIONE
----------------
1. Clonare o scaricare il file `working_main_xauusd_ultimate.py` in una cartella.
2. Aprire un terminale nella cartella del bot.
3. Creare un ambiente virtuale (consigliato):


6. CONFIGURAZIONE OBBLIGATORIA
------------------------------
**IMPORTANTE:** Modificare i parametri di connessione e trading all'interno del file
            Python prima di eseguire il bot.

Aprire `working_main_xauusd_ultimate.py` e modificare la classe `Config`:

```python
class Config:
 # ⚠️ INSERIRE I PROPRI DATI MT5 (DEMO O REAL)
 MT5_ACCOUNT = 333333333          # <-- Sostituire con proprio login
 MT5_PASSWORD = "password"        # <-- Sostituire con propria password
 MT5_SERVER = "Server-MT5"      # <-- Sostituire con il server del broker

 SYMBOL = "XAUUSD.s"              # <-- Verificare il suffisso del simbolo (.s, .m, etc.)

 # Parametri di trading
 FIXED_LOT_SIZE = 0.01            # Lotto fisso (si può cambiare)
 MANUAL_LOT_SIZE = 0.01           # Lotto manuale (cambiare qui per modificare il lotto)

 MIN_CONFIDENCE = 0.70            # Soglia confidenza ML per aprire trade, consigliata 0.70
 MAX_SPREAD_PIPS = 30.0           # Spread massimo in pip per XAUUSD

 # Gestione del rischio
 MAX_DAILY_LOSS_PERCENT = 2.5
 MAX_CONSECUTIVE_LOSSES = 3
 MAX_DRAWDOWN_PERCENT = 5.0
```
7.  COME FUNZIONA IL SISTEMA IBRIDO DI TAKE PROFIT
------------------------------
Il bot calcola il TP combinando fino a 4 strategie con pesi dinamici:
Strategia	Descrizione	Peso iniziale
ATR_BASED	Distanza TP = ATR * moltiplicatore (variabile in base al regime di mercato)	25%
ML_PREDICTED	TP predetto da un modello di regressione addestrato su dati storici	35%
SUPPORT_RESISTANCE	TP posizionato al livello di resistenza/supporto più vicino	25%
RL_OPTIMIZED	TP aggiustato da un agente di Reinforcement Learning (Q‑learning)	15%

Dopo ogni trade chiuso, i pesi vengono aggiornati in base al profitto realizzato,
premiando le strategie che hanno contribuito a trade vincenti.

    AVVERTENZE E DISCLAIMER

⚠️ RISCHIO DI PERDITA FINANZIARIA
Il trading di strumenti finanziari comporta un alto rischio e non è adatto a tutti
gli investitori. I risultati passati non garantiscono performance future.
Questo bot è fornito a scopo educativo e di ricerca. L'utente è l'unico
responsabile delle decisioni di trading e delle eventuali perdite subite.

⚠️ UTILIZZARE SOLO SU CONTO DEMO INIZIALMENTE
Testare il bot per almeno alcune settimane su un conto demo prima di considerare
l'utilizzo su conto reale.

⚠️ CREDENZIALI HARDCODED
Le credenziali MT5 sono scritte in chiaro nel file di configurazione.
Non condividere il file con terzi e proteggere adeguatamente il proprio computer.
