# 🚀 MATCH Vagas × Candidatos

🔹 Plataforma inteligente para seleção de talentos que combina clustering, classificadores empilhados e similaridade semântica.

---

## 🔎 Visão Geral

📝 Funcionalidades principais:

* 📌 **Cadastro de Vagas**: título, cliente, descrição, competências, indicador SAP.
* 👤 **Cadastro de Candidatos**: currículo, experiência, formação, idiomas, indicador SAP.
* 🧩 **Clustering**: agrupa perfis semelhantes via KMeans.
* 🏆 **Ranking Top‑10**: combinação de classificadores e SBERT.

---

## 🛠️ Arquitetura do Projeto

```
📁 root/
├─ 📂 ml_api/            # Backend REST API
│   ├─ 📂 app/           # Código da aplicação
│   ├─ 📂 data/          # CSVs e DB (SQLite)
│   ├─ 📂 model/         # Modelos serializados
│   ├─ 📂 tests/         # Testes Padronizados para validar aplicação
│   └─ 🗒️ main.py        # Inicialização do Flask
└─ 📂 web_dash/          # Dashboard Web
    ├─ 🗒️ app.py         # Frontend Flask
    ├─ 📂 templates/     # Páginas HTML
    └─ 📂 static/        # CSS e assets
```

---

## 🤖 Modelos de Machine Learning

1. **🔸 Clustering (KMeans) Vagas & Candidatos**

   * Agrupa perfis similares para pré-seleção.
   * Facilita correspondência inicial entre perfis similares.
   * Realiza agrupamento de perfis otimizando proximidade textual e técnica.


2. **🔸 Classificadores Otimista & Pessimista**

   * Stacking: CatBoost + RandomForest + LightGBM.
   * Cada classificador é um StackingClassifier (CatBoost, RandomForest, LightGBM).
   * O modelo otimista enfatiza potenciais positivos, enquanto o pessimista pende à cautela.
   

3. **🔸 Similaridade Semântica (SBERT)**

   * Embeddings Transformers para captar nuances textuais e medir afinidade vaga‑candidato.

4. **⚖️ Cálculo de Compatibilidade**

   * `base = (s_otimista + (1 - s_pessimista)) / 2`
   * `compatibilidade = 0.7*base + 0.3*sim_sbert`
   * Peso de 70 % para decisões de negócio (classificadores) e 30 % para enriquecimento semântico.



### 🎯 Interpretação das Métricas de Produção

- **Modelo OTIMISTA**  
  - **F1‑Score (0.8930)**: Excelente equilíbrio entre precisão e recall.  
  - **Precision (0.8111)**: Em 81.11% das vezes, quando o modelo prevê “match”, o candidato realmente combina.  
  - **Recall (0.9932)**: Captura quase todos os casos positivos, garantindo cobertura quase completa.  
  - **Threshold (0.24)**: Ponto de corte ajustado em 24% de probabilidade para classificar como “match”.  
  - **Avaliação**: **Excelente** para a primeira triagem, priorizando recall alto sem sacrificar muito a precisão.

- **Modelo PESSIMISTA**  
  - **F1‑Score (0.4101)**: Equilíbrio moderado entre precision e recall, indicando melhoria significativa.  
  - **Precision (0.3569)**: Em 35.69% das vezes, uma previsão de “match” é correta.  
  - **Recall (0.4818)**: Identifica aproximadamente 48.18% dos candidatos realmente compatíveis.  
  - **Threshold (0.51)**: Corte em 51% de probabilidade, tornando o critério de “match” mais seletivo.  
  - **Avaliação**: **Bom** — É um filtro confiável para reduzir falsos positivos, devendo ser usado em sinergia com o modelo otimista.

---

🔍 **Resumo Geral**:  
- O **modelo otimista** garante **quase nenhuma perda** de candidatos relevantes (recall ≈ 99.3%) com **alto nível de acerto** (precision ≈ 81%).  
- O **modelo pessimista**, com F1 ≈ 41%, tornou‑se **útil** para confirmar e reforçar a qualidade do match, equilibrando riscos e retirando da análise os falso positivo.  

✅ Esses resultados mostram que a combinação das duas visões (otimista e pessimista) gera um sistema de seleção robusto, maximizando cobertura e confiabilidade, eliminando os falso positivos.


---

## 📋 Como Usar a API

1. **Iniciar Servidor**

   ```bash
   cd ml_api
   flask run --host=0.0.0.0 --port=5000
   ```

2. **Cadastrar Vaga**

   ```bash
   curl -X POST http://localhost:5000/vagas/ \
     -H "Content-Type: application/json" \
     -d '{ "titulo":"Data Scientist", "cliente":"Empresa X", "descricao":"Python, ML e AWS", "competencias":"Python ML AWS", "eh_sap":0 }'
   ```

3. **Cadastrar Candidato**

   ```bash
   curl -X POST http://localhost:5000/candidatos/ \
     -H "Content-Type: application/json" \
     -d '{ "nome":"João Silva", "email":"joao@exemplo.com", "cv_pt":"Experiência...", "informacoes_profissionais":{...}, "formacao_e_idiomas":{...}, "eh_sap":0 }'
   ```

4. **Obter Ranking Top‑10**

   ```bash
   curl http://localhost:5000/match/<vaga_id>
   ```

Retorno JSON: lista de candidatos com `{ id, nome, email, compatibilidade (%) }`.

5. **Teste Postman**
    Arquivo de teste no caminho: ml_api\tests\Postman\MATCH API Tests.postman_collection.json

    5.1. **Importar a coleção**  
   - Abra o Postman.  
   - Clique em **Import** (no canto superior esquerdo).  
   - Selecione o arquivo `MATCH API Tests.postman_collection.json`.

    5.2. **Configurar variáveis de ambiente**  
   - Crie ou selecione um *Environment* chamado `local`.  
   - Defina a variável `base_url` com o valor:
     ```
     http://localhost:5000
     ```

    5.3. **Executar os testes**  
   - Vá em **Collections**, expanda **MATCH API Tests**.  
   - Clique em **Run** (Collection Runner).  
   - Escolha o Environment `local` e clique em **Start Run**.  

    5.4. **Verificar resultados**  
   - O Postman exibirá, para cada requisição, o status code e eventuais scripts de validação (asserts).  
   - Todos os endpoints (`POST /vagas`, `GET /vagas`, `POST /candidatos`, `GET /candidatos`, `GET /match/:id`) serão testados automaticamente.  

---

## 💾 Deploy & Configuração

1. Configurar `ml_api/app/config.py` (DB\_PATH, MODEL\_DIR).
2. Gerar banco SQLite:

   ```bash
   python ml_api/model/db_create/create_db_sqlite.py
   ```
3. Executar API e Dashboard.

## 🐳 Executando com Docker
Para rodar API e WebDash via Docker Compose:

1. **Gerar as imagens**  
   No terminal (VSCode, PowerShell, CMD ou bash), na raiz do projeto:
   ```bash
   docker build -t match-api -f ml_api/Dockerfile .
   docker build -t match-web -f web_dash/Dockerfile .

2. **Gerar as imagens**  
    docker run -d --name match-api -p 5000:5000 match-api
    docker run -d --name match-web -p 8000:8000 match-web

3. **Acessar**
    API: http://localhost:5000
    Dashboard Web: http://localhost:8000
    ⚙️ O mesmo requirements.txt é utilizado pelas duas imagens e deve estar na raiz do projeto ao lado do docker-compose.yml.

---

## 📞 Contato
Nathan Rafael Pedroso Lobato
✉️ nathan.lobato@outlook.com.br

---

## 📄 Licença
Este projeto está sob a licença MIT — livre para uso, modificação e distribuição.
