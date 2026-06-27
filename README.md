# 🧠 Unsupervised Customer Segmentation Dashboard

A full-stack interactive ML web dashboard that segments customers
into distinct behavioral groups using THREE machine learning
techniques — Unsupervised Learning, Association Rules, AND
Reinforcement Learning combined in one project.

Built under the mentorship of **Mr. Rubeshkannaravichandran** and
the visionary leadership of **Mr. Senthil Rajamarthandan**,
Founder of DigiDARA Technologies Private Limited. 🙏

---

## 🎯 Objective
To group 1,000 customers into 5 distinct behavioral segments
and dynamically refine marketing strategies using Q-Learning.

---

## 🛠️ Tech Stack

| Layer | Tools Used |
|-------|-----------|
| Clustering | Scikit-learn (K-Means + PCA) |
| Association Rules | MLxtend (Apriori) |
| Reinforcement Learning | Q-Learning (custom implementation) |
| Data Processing | Python · Pandas · NumPy |
| Visualization | Plotly (interactive charts) |
| Backend | Flask |
| Frontend | Bootstrap + Custom CSS + JavaScript |

---

## 🤖 Three ML Techniques Combined

### 1️⃣ K-Means Clustering
- Groups 1,000 customers into **5 segments**
- Features: **Age · Annual Income · Spending Score**
- PCA applied for 3D visualization

### 2️⃣ Apriori Association Rules
- Uncovers **hidden patterns** within demographic categories
- Results exported to `models/rules.csv`

### 3️⃣ Q-Learning (Reinforcement Learning)
- Dynamically **refines segment strategies** over time
- Agent learns optimal marketing actions per segment
- Policy & Q-table saved as `policy.pkl` & `q_table.pkl`

---

## 📊 Dashboard Features

| Section | What it shows |
|---------|--------------|
| 🌐 3D Cluster Projection | All 5 segments in 3D space |
| 📦 Box Plot Distribution | Age, Income & Spending Score per segment |
| 💰 Income vs Spending | Relationship between income and spending |
| 👥 Gender Distribution | Demographic breakdown per cluster |
| 🏆 Radar Chart | Algorithm evaluation & model performance |
| 📤 Export | Download segment profiles for business use |

---

## 📁 Project Structure

```
unsupervised_customer_segmentation/
│
├── app.py                          ← Root Flask entry point
├── backend/
│   ├── app.py                      ← Backend logic
│   ├── clean_data.py               ← Data cleaning pipeline
│   └── train_qlearning.py          ← Q-Learning training
├── frontend/
│   ├── index.html                  ← Dashboard UI
│   ├── index.css                   ← Styling
│   └── dashboard.js                ← Interactive logic
├── models/
│   ├── kmeans_model.pkl            ← Trained K-Means model
│   ├── kmeans.pkl                  ← K-Means backup
│   ├── pca.pkl                     ← PCA model
│   ├── scaler.pkl                  ← Scaler
│   ├── apriori_rules.pkl           ← Apriori model
│   ├── rules.csv                   ← Association rules output
│   ├── policy.pkl                  ← RL policy
│   └── q_table.pkl                 ← Q-Learning table
├── data/
│   └── Cleaned_mall_customer_data.csv
├── static/
│   ├── background.jpeg
│   └── cluster_distribution.png
├── models_creation.py              ← Model building script
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/unsupervised-customer-segmentation.git

# 2. Navigate into the folder
cd unsupervised-customer-segmentation

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

# 5. Open in browser
http://localhost:5000
```

---

## 🔮 Future Enhancements
- [ ] Add DBSCAN & Hierarchical Clustering for comparison
- [ ] Deploy live on cloud (AWS / Render)
- [ ] Real-time data input & live dashboard updates

---

## 🙏 Acknowledgements
- **Mr. Rubeshkannaravichandran** — Mentor & Guide
- **Mr. Senthil Rajamarthandan** — Founder, DigiDARA Technologies Pvt. Ltd.

---

## 🌐 Live Demo
🎥 [Watch Demo Video] https://youtu.be/Hb3y8CDRgXk

---
## 👩‍💻 About Me
Sasmita.S Ai engineer 🔗 [LinkedIn] www.linkedin.com/in/sasmitasenthil
