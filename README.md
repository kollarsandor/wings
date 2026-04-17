



markdown
 wings – Teljes dokumentáció (Magyar)
> Forrás: https://deepwiki.com/kollarsandor/wings/
> Generálva: 2026-04-18

---

 Overview

A Neural Network Framework egy többnyelvű deep learning rendszer, amelyet arra terveztek, hogy specializált programozási nyelvek előnyeit használja ki a machine learning életciklusának különböző szakaszaiban. A rendszer alrendszerekre bontásával – a magas szintű adat-orkesztrációtól az alacsony szintű SIMD-gyorsított kernelekig – a projekt egyensúlyt teremt a fejlesztői produktivitás és a nyers számítási teljesítmény között.

 Cél és hatókör

A framework egy teljes pipeline-t biztosít neurális modellek betanításához, szerializálásához és kiszolgálásához (serving). A monolitikus rendszerekkel ellentétben a „legjobb eszközt az adott feladatra” megközelítést alkalmazza:
- Zig kezeli az alapvető ütemezést és a backpropagation engine-t.
- Inko felel a konkurens adatbetöltésért (data loading) és a bináris szerializálásért.
- Lobster biztosítja a serving layert és a kérések útválasztását (routing).
- Simit implementálja a gráf-alapú számításokat és a fizikai szimulációkat.
- Scopes tensor runtime-ként és compute graph menedzserként működik.

 Multi-Language Architektúra

Az architektúra több specializált környezet között oszlik el. Minden nyelvet az egyedi memóriakezelési, konkurens végrehajtási vagy matematikai kifejezési erősségei alapján választottak ki.

 Futtatás és fejlesztés

A projektet elsődlegesen a Zig belépési ponton keresztül futtatják, amely bemutatja az alapvető matematikai és ütemezési (scheduling) komponenseket. 
A framework inicializálható és tesztelhető a Zig fordító segítségével, például a zig run main.zig paranccsal, de a specializált komponensek (Inko, Lobster) futtatásához a saját eszközkészletükre is szükség van.

---

 Getting Started & Environment Setup

Ez a szakasz a fejlesztői környezet beállításához nyújt technikai utasításokat. A többnyelvű jelleg miatt a környezet specifikus fordítókat és futtatókörnyezeteket igényel a Zig, Inko, Lobster, Simit, Scopes és ISPC számára.

 Gyors indítás Repliten keresztül

A projekt úgy van konfigurálva, hogy azonnal futtatható legyen egy Replit környezetben a biztosított .replit konfiguráció használatával. A környezet inicializálása a nix csomagkezelőt használja a megfelelő modulok betöltéséhez, míg a végrehajtás párhuzamos munkafolyamatokat indít el.

 Helyi környezet beállítása

A framework Repliten kívüli futtatásához az alábbi eszközkészletek (toolchain-ek) szükségesek:
1. Zig (Core Engine): Az alapvető logika (backprop, inference, scheduling). Szükséges verzió: 0.11.x.
2. Inko (Data Pipeline): Erős konkurens modellje miatt adatbetöltésre és modellek szerializálására szolgál.
3. Lobster (Serving): A ServingServer és a RequestRouter megvalósításához.
4. Simit & Scopes (Compute): A Simit a fizikai szimulációkhoz és GNN-ekhez, míg a Scopes a magas szintű tensor runtime-hoz szükséges.

 Implementációs részletek: Zig Demo

A demó funkció (Learning Rate Simulation) bemutatja a bemelegítési (Warmup) és a koszinuszos csökkenési (Cosine Annealing) stratégiákat. Különféle optimalizált tensor primitívek állnak rendelkezésre, beleértve a ReLU aktivációt és a Softmax normalizációt a stabil valószínűségszámításhoz.

---

 Multi-Language Architecture

A rendszer öt elsődleges, nyelvek által vezérelt területre oszlik, amelyeket bináris szerializációs protokollok és megosztott memóriamegoldások hangolnak össze.

- Zig (Core Engine): A legteljesítmény-kritikusabb részeket kezeli, mint például az alacsony szintű ütemezés, a forward és backward pass végrehajtása zéró többletköltséggel.
- Inko (Data & I/O): A ParallelDataLoader révén masszív adat-augmentációt és lemez I/O műveleteket végez a fő számítási szálak blokkolása nélkül. A keretrendszer saját bináris checkpoint formátumát is ez kezeli.
- Lobster (Serving): A kérés-sorok (request queues) és az aszinkron modell-regisztrációk (model registry) flow-alapú kezelésére szolgál.
- Scopes (Runtime): Magas szintű absztrakciókat biztosít a Tensor műveletekhez és a ComputeGraph felépítéséhez, áthidalva a Zig motor és az optimalizálók (pl. AdamW) matematikai definícióit.
- Simit (Physics/Graph): Specializált számítási terhelések, például gráf neurális hálózatok (GNN) és folyadékszimulációk végrehajtása.

A nyelvek közötti kommunikáció elsősorban egyedi NFML bináris formátumon, illetve C-ABI memóriamegosztáson (Zig és Scopes között) alapszik.

---

 Training Pipeline

A betanítási (training) pipeline egy elosztott, többnyelvű munkafolyamat, amely a nyers adatokat a tárolókból magas teljesítményű számítási kerneleken keresztül az architektúra frissítéséig vezeti.

 Végponttól végpontig tartó munkafolyamat
1. Adatbefogadás (Data Ingestion): A ParallelDataLoader párhuzamosan lekéri és augmentálja a nyers mintákat.
2. Tokenizálás (Tokenization): A szöveg számokká alakítása a ParallelTokenizer segítségével.
3. Forward Pass: A BackpropEngine végrehajtja a predikciókat a Layer stacket alkalmazva.
4. Veszteség számítás (Loss): A predikciók és a célértékek (targets) különbségének kiszámítása.
5. Backpropagation: A gradiensek visszaterjesztése a ComputationGraph-on keresztül.
6. Optimalizáció (Optimization): A BaseOptimizer frissíti a súlyokat a számított gradiensek alapján.
7. Ütemezés (Scheduling): A LearningRateScheduler beállítja a hiperparamétereket a következő lépéshez.

---

 Data Loading & Preprocessing

Ez az alrendszer egy magas teljesítményű, párhuzamosított pipeline, amely Inko nyelven íródott. Célja, hogy különválassza az adatlekérést, az augmentációt és a batch-képzést a fő betanítási ciklustól.

 Párhuzamos adatbetöltő architektúra
A ParallelDataLoader aszinkron üzenetküldést használ, egyedi munkavégzők (worker) felé delegálva a feladatokat. Minden worker saját kontextusban fut (elkerülve a Global Interpreter Lock problémáit), és egy LRU (Least Recently Used) gyorsítótárat (SampleCache) használ a gyakori adatokhoz, így minimalizálva a lemez I/O terhelést.

 Dinamikus kitöltés (Padding)
Amikor a CollateFunction egy Batch-et hoz létre a változó hosszúságú mintákból (ami NLP esetében gyakori), dinamikus paddinget végez a köteg leghosszabb mintájához igazítva, miközben megfelelő figyelem-maszkot (attention_mask) is generál.

---

 Tokenizer

A tokenizációs alrendszer a nyers természetes nyelvű szöveg diszkrét egész számok sorozatává (tokenekké) történő átalakításáért felelős. Olyan algoritmusokat támogat, mint a Byte Pair Encoding (BPE), a WordPiece és a Unigram.

A ParallelTokenizer a hatalmas adathalmazok feldolgozására worker-alapú (munkavégző) párhuzamosítási modellt alkalmaz. A folyamat lépései:
1. Pre-Tokenization: Normalizálja a szöveget, szétválasztja a karaktereket (külön kezelve pl. a CJK nyelveket).
2. Encoding: Az alkalmazott algoritmussal a szöveget al-szavakra (subword) bontja a meglévő szótár (Vocabulary) segítségével.
3. Post-Processing: Speciális markerek (pl. [BOS], [EOS], [PAD]) hozzáadása, csonkítás (truncation).

---

 Backpropagation Engine

A Zig nyelvben implementált BackpropEngine kezeli a tensorok, rétegek, valamint a forward és backward pass végrehajtását. A rendszer a gradient checkpointingot és a kevert precíziós (mixed-precision) horog-műveleteket is támogatja.

 Alapvető adatszerkezetek
- TensorDescriptor: A nyers adatokat, a formai (shape) információkat és a gradienseket tároló egység. 
- LayerConfig: Réteg paraméterek konfigurálása (kernel méret, stride, dropout).
- LayerCache: Köztes aktivációk tárolása a backward pass-hoz.

A backprop motor a forward fázisban iterál a gráf csomópontjain, frissíti az aktivációkat. A backward pass során az Autodiff láncszabályt (chain rule) alkalmazva halad visszafelé, skálázza a gradienseket (grad_scaling) a pontosságvesztés elkerülése végett, és norm-clippinget végez, megelőzve a gradiens-robbanást. A paraméterek inicializálása alapvetően a Kaiming (He) és Xavier (Glorot) módszereket alkalmazza.

---

 Automatic Differentiation (Gradient Compute)

A Lobster nyelven megvalósított rendszer támogatja mind a forward-mode, mind a reverse-mode differenciálást. A számítási gráf (ComputationGraph) felel az elvégzett műveletek naplózásáért (GradientTape segítségével).

- Variable & TensorShape: Nyomon követik a metaadatokat (hogy levél csomópont-e az adott tenzor, és szükséges-e számára gradiens).
- GradientTape: Iteratívan menti az operációk adatait, amikor a rögzítés (is_recording) aktív.
- A reverse-mode (backward) logikája kiszámítja a Jacobi-mátrixokat minden egyes OperationType-hoz, a mentett cache (pl. ReLU esetén $\le 0$ ellenőrzés) alapján osztja szét a gradienseket a bemenetekre. Ezenkívül a framework támogatja az akkumulált gradienseket (GradientAccumulator), ami nagyobb virtuális batch méreteket tesz lehetővé.

---

 Optimizers

Az optimalizálók frameworkje a modell paramétereinek gradiens alapú frissítését kezeli. A BaseOptimizer központilag gyűjti össze a ParameterGroup és ParameterState objektumokat.

Támogatott optimalizáló változatok:
- SGD: Alapvető algoritmus momentum és Nesterov-gyorsítás támogatással.
- Adam / AdamW: Az Adam az adaptív momentum becslését végzi, míg az AdamW szétválasztja a weight decay logikát a gradiens másodlagos nyomatékától, így hatékonyabban bünteti a nagy súlyokat.
- LAMB, LARS, Adafactor: Speciális algoritmusok extrém nagy batch méretekhez és alacsony memóriafelhasználáshoz.

A step() fázis minden iterációban lekéri az optimalizálási konfigurációkat és az eltárolt csúszóátlagokat (exp_avg, exp_avg_sq), és frissíti a tényleges paraméter tenzorokat.

---

 Learning Rate Scheduling

A tanulási ráta (learning rate) időbeli, dinamikus skálázásáért a Zig nyelven írt LearningRateScheduler felel. Támogatja az alap algoritmusokat és a ciklikus megközelítéseket is.

Támogatott stratégiák:
- Step / Multi-Step: Fix lépésközönként csökkenti a tanulási rátát egy adott tényezővel (gamma).
- Cosine Annealing: Folyamatos koszinusz görbét követve csökkenti a rátát a bázis értékről egy minimum felé.
- One-Cycle Policy / SGDR: Lehetővé teszi a ciklikus iterációt és a meleg újraindításokat (warm restarts).
- ReduceOnPlateau: A validációs metrikát figyeli, és "türelem" (patience) periódus letelte után bünteti a rátát, ha nincs javulás.

---

 Inference & Serving

A dedikált Inference és Serving alrendszer két rétegben dolgozik. A magas szintű Serving Layer (Lobster) a hálózati kérésekért (request routing) és a kötegelésért (batch accumulator) felelős, míg az alacsony szintű Inference Engine (Zig) a memóriaszintű tenzorkezelésért, a kvantálásért és a gyors predikciókért felel.

 Inference Engine (3.1)
- Stride-alapú indexelés: A tenzorok multidimenzionális adatait egy folyamatos memóriablokkra (Array_f32) képezi le offset logikák alkalmazásával.
- KV Cache: Transformer modellek autoregresszív generálása esetén eltárolja az előző kulcsokat és értékeket, megelőzve az újra-számítást.
- QuantizedTensor: Int8 és Int4 csoportos kvantálást (group quantization) biztosít a RAM takarékosság és a megnövelt throughput érdekében.
- Platform optimalizálás: Dinamikusan érzékeli a processzor AVX2, AVX512 vagy NEON támogatását a futásidejű kernelek futtatásához.

 Serving Layer (3.2)
- A modell-menedzsmentet a ModelRegistry oldja meg, biztosítva a betöltött modellek metadatáinak és memóriafoglalásának elkülönítését.
- A beérkező REST/hálózati kéréseket egy aszinkron RequestQueue puffereli.
- A BatchAccumulator dinamikusan optimalizálja, hogyan fűzi össze a kéréseket hardveres batch-é (várva egy timeoutig, vagy amíg a köteg megtelik).
- Tartalmaz automatikus RateLimiter-t, HealthChecker-t és valós idejű telemetriát (ServerMetrics), monitorozva a throughput-ot (EPS) és a P99-es latenciát.

---

 Runtime & Compute Backends

A low-level rétegben a keretrendszer dual-backend (kettős háttérrendszer) modellt használ a compute műveletekhez.

 Scopes Runtime (4.1)
A Scopes felel a magas szintű grafikus logika értelmezéséért. Itt él a Tensor data struktúra, és a layer forward-pass diszpécsere (Layer Dispatch). Szintén itt lettek natívan implementálva az aktivációs függvények (ReLU, GELU, SiLU, Softmax) és a LayerNorm/RMSNorm logikák.

 SPMD Vectorization - ISPC (4.2)
Az optimális CPU teljesítményhez a keretrendszer az Intel ISPC (Implicit SPMD Program Compiler) modelljét integrálja.
- Tiled GEMM: A mátrixszorzásokhoz gyorsítótár-lokalitást javító mozaikszerű blokkolást (64x64) használ a matmul_blocked eljárás.
- Gyors közelítések (Transcendental Approximations): A drága matematikai hívásokat (pl. exponentiális fast_exp, tangens fast_tanh, inverz gyök fast_rsqrt) olyan algoritmusokkal közelíti (pl. Newton iteráció vagy Taylor sorok mentén), melyek a precizitás apró feladásával óriási teljesítményt adnak.
- Kiemelt kernels támogatja a dedikált Transformer operációkat, köztük a Rotary Positional Embedding (RoPE) alkalmazását.

---

 Graph Computation & Physics Simulation

A Simit nyelven futó fizika motor (Physics Engine) elosztott feladatként dolgozza fel a folytonos fizikai elemeket (csomópontok, élhálózatok) és biztosítja a gráf neurális hálók (GNN) hátterét.

 Graph Neural Network Layers (5.1)
A modul a gráf-strukturált adatokhoz speciális elemeket (NeuralVertex, NeuralEdge) definiál.
- Támogatja a Graph Attention (GAT) rétegeket skálázott dot-product megoldásokkal, valamint általános üzenetátviteli (Message Passing) metódusokat (MPNN, GCN).
- Gradiensek nyilvántartása csomóponti (vertex) és él (edge) szinteken zajlik.

 Physics Simulation Kernels (5.2)
- Tömeg-rugó (Mass-Spring) rendszer szimuláció Hooke törvényeinek és a mechanikai csillapításnak a beépítésével.
- Térfogati és felületi megőrzés, hálós (mesh) műveletek (háromszögek terület- és normálszámítása, tetraéderek térfogat integrálása).
- Folyadékdinamika, ami Smoothed Particle Hydrodynamics (SPH) kerneleken alapszik a sűrűség, nyomás és viszkozitás szimulálására.

---

 Memory Management

A keretrendszer memóriakezelése a töredezettség és a szemétgyűjtési költségek (overhead) ellen küzd egy robusztus, több szintből álló allokációs paradigmával.

A fő szálért felelős a GlobalMemoryManager, amely a kéréseket további al-menedzserekhez rendeli:
- MemoryPool: Fix méretű blokkok számára (bitképek alapján gyors allokáció O(1) idővel).
- Arena: Task-scoped műveletekhez (marker alapú egyszeri és tömeges felszabadítás bump-allokátorral).
- GradientMemoryPool: Specifikusan a forward/backward pass gradienseire fókuszálva (könnyű réteg szintű ürítés a model creep elkerüléséhez).
- CacheAlignedAllocator: Biztosítja, hogy a tenzorok data pointerei 64-bájtosan (cache vonal mentén) igazodva legyenek, elkerülve a 'false sharing' problémákat a CPU szálak között.
A memóriaszivárgások ellenőrzésére egy AllocationTracker ad telemetriai adatokat, checksumot és verem-visszakövetést (stack trace) mentve.

---

 Model Serialization & Deployment

A betanított hálózatok lementéséért (checkpointing) és külső integrációjáért felelős modul egy speciális bináris kódolást használ. 

 NFML Bináris Formátum
A keretrendszer formátuma (Magic byte: 0x4E464D4C) memórialeképezésre (memory-mapping) van optimalizálva. A fájl tartalmaz:
1. Fejlécet és verzióazonosítót.
2. JSON alapú globális metadata szakaszt (rétegek és topológia leírása).
3. Tenzor deskriptorokat és alakokat.
4. Nyers tenzor buffereket (64-bájtosan igazítva a SIMD-hez).

Az eszközkészlethez hozzátartozik a ModelExporter, amely biztosítja a Safetensors integrációt Python környezetekhez, valamint opciót kínál az Int8/Float16 menetközbeni export-alapú kvantálására.

---

 Glossary (Szójegyzék)

- BackpropEngine: A training (betanítási) szakasz központi vezérlője, megépíti a gráfokat és kezeli a checkpointokat (Zig).
- ComputationGraph: Irányított aciklikus gráf (DAG) a tenzorműveletek Autodiff-jéhez (Lobster).
- NFML: Neural Framework Markup Language, a keretrendszer 64-bájt illesztett, little-endian bináris formátuma.
- KV Cache: Key-Value gyorsítótár az autoregresszív iterációk spórolásához.
- ISPC & SPMD: Single Program, Multiple Data alapú vektorizált futtatás az Intel compilerén keresztül.
- SPH: Smoothed Particle Hydrodynamics a fizika részecske folyadék modellezéséhez.
- RoPE: Rotary Positional Embedding; forgatási alapú pozíciós kódolás a transformer layer-ekben.

