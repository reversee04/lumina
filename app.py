"""
LUMINA — Flask Web Application v2.0
Enhanced with /score_user and /match_jobs API endpoints
"""

import os
import math
import re
import json
import uuid
from datetime import datetime

from flask import Flask, render_template_string, request, jsonify

# ML / numeric imports
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ── Training / Opportunity Data ─────────────────────────────

TRAINING_PROFILES = [
    ("Built fintech app 2400 users mobile money API 63 github commits payment system mentored 15 students", 8, 3, 85, "Gold"),
    ("Designed brand identities 12 startups 47 freelance projects 4.9 rating ui kit downloaded 8000 times motion graphics 320k subscribers", 7, 4, 82, "Gold"),
    ("Solar microgrid 23 rural households arduino iot energy monitoring 4 ngos trained 8 apprentices reduced costs 60 percent", 6, 5, 78, "Gold"),
    ("Machine learning kaggle top 15 percent sales forecasting saved 180k demand prediction upwork dashboards 6 clients", 7, 2, 76, "Gold"),
    ("Whatsapp bot 340 farmers market prices precision farming drip irrigation 35 percent yield increase 60 farmers trained grants", 6, 4, 74, "Gold"),
    ("Wordpress websites 3 clients logo design canva social media posts basic html css", 3, 1, 35, "Bronze"),
    ("Data entry excel spreadsheets pivot tables basic analysis 2 years experience", 2, 2, 28, "Bronze"),
    ("Repair mobile phones 5 years customers hardware technician basic soldering", 3, 5, 40, "Bronze"),
    ("Python scripts automation 1 project github portfolio beginner level tutorials coursera", 3, 1, 32, "Bronze"),
    ("React developer portfolio 3 projects e-commerce admin dashboard api integration", 5, 2, 55, "Silver"),
    ("Graphic designer 200 projects fiverr logo branding social media print vector", 5, 3, 58, "Silver"),
    ("Network technician cisco routers setup 50 installations troubleshooting helpdesk", 4, 4, 52, "Silver"),
    ("Video editor youtube 50k subscribers after effects premiere adobe suite color grading", 5, 3, 60, "Silver"),
    ("Agricultural extension officer soil testing crop rotation 200 farmers trained", 5, 6, 62, "Silver"),
    ("Full stack developer node react postgres aws deployment ci cd 10 production apps", 9, 4, 88, "Gold"),
    ("Open source contributor 200 commits popular library npm package 10k downloads", 8, 3, 80, "Gold"),
    ("Embedded systems stm32 pcb design firmware manufacturing prototype 5 products", 7, 5, 79, "Gold"),
    ("UX researcher 15 usability studies 2 published case studies product redesign doubled retention", 7, 4, 81, "Gold"),
    ("Taught coding 300 students bootcamp curriculum designed online course 2000 enrolled", 6, 5, 77, "Gold"),
    ("SEO content writing 50 articles ranked page 1 google 200k organic traffic", 5, 3, 56, "Silver"),
    ("3d printing product design cad fusion360 prototypes sold etsy 400 units", 5, 2, 54, "Silver"),
    ("Bookkeeping quickbooks small business 20 clients tax preparation certified", 4, 4, 50, "Silver"),
]

OPPORTUNITY_DATABASE = [
    {"title": "Remote React Developer", "source": "Toptal", "domain": "software development react javascript frontend web app component api", "pay": "$35-55/hr", "type": "Freelance"},
    {"title": "Python Data Analyst", "source": "Upwork", "domain": "python data analysis pandas sql machine learning statistics visualization", "pay": "$25-40/hr", "type": "Freelance"},
    {"title": "UI/UX Designer", "source": "Fiverr Pro", "domain": "figma design ux ui wireframe prototype user research branding", "pay": "$20-45/hr", "type": "Freelance"},
    {"title": "Solar Energy Tech Consultant", "source": "GOGLA Network", "domain": "solar energy renewable electricity hardware installation grid off-grid iot", "pay": "$18-30/hr", "type": "Contract"},
    {"title": "AgTech Innovation Fellow", "source": "Acumen Fund", "domain": "agriculture farming crops yield data sensors irrigation food security", "pay": "$1,500/mo stipend", "type": "Fellowship"},
    {"title": "Full Stack Engineer", "source": "Remote.com", "domain": "full stack node express postgres mongodb api backend frontend deployment", "pay": "$40-60/hr", "type": "Full-time Remote"},
    {"title": "Brand Identity Designer", "source": "99designs", "domain": "brand logo identity visual design illustrator typography color creative", "pay": "$500-2000/project", "type": "Project"},
    {"title": "IoT Systems Developer", "source": "Freelancer", "domain": "iot arduino raspberry pi embedded firmware sensors hardware electronics", "pay": "$22-38/hr", "type": "Freelance"},
    {"title": "Digital Finance Specialist", "source": "CGAP / World Bank", "domain": "fintech mobile money payment api banking inclusion financial digital", "pay": "$3,000/mo", "type": "Consultancy"},
    {"title": "Open Source Contributor Grant", "source": "GitHub Sponsors", "domain": "open source github javascript python library contributor developer", "pay": "$500-2000/mo", "type": "Grant"},
    {"title": "Data Science Intern (Remote)", "source": "UNICEF Giga", "domain": "data science machine learning python analytics visualization impact", "pay": "$1,200/mo", "type": "Internship"},
    {"title": "Community Tech Educator", "source": "Andela Learning", "domain": "teaching coding mentoring curriculum bootcamp training students education", "pay": "$1,800/mo", "type": "Full-time"},
    {"title": "Graphic Design Freelancer", "source": "PeoplePerHour", "domain": "graphic design photoshop illustrator print digital poster social media", "pay": "$15-30/hr", "type": "Freelance"},
    {"title": "Climate Tech Innovation Grant", "source": "Ashoka Changemakers", "domain": "climate renewable energy environment sustainability innovation impact", "pay": "$5,000 grant", "type": "Grant"},
    {"title": "Backend Node.js Developer", "source": "Arc.dev", "domain": "backend nodejs express api rest microservices database postgres mongo", "pay": "$38-52/hr", "type": "Freelance"},
    {"title": "Video & Motion Designer", "source": "Envato Studio", "domain": "video editing motion graphics after effects premiere animation youtube", "pay": "$20-40/hr", "type": "Freelance"},
    {"title": "Smallholder AgTech Advisor", "source": "One Acre Fund", "domain": "smallholder agriculture farmer training soil crop yield food malawi kenya", "pay": "$1,400/mo", "type": "Contract"},
    {"title": "Cybersecurity Analyst (Junior)", "source": "HackerOne", "domain": "security network cybersecurity vulnerability pentesting bug bounty", "pay": "$15-35/hr", "type": "Freelance"},
    {"title": "Mobile App Developer", "source": "Clutch.co", "domain": "mobile react native flutter android ios app development", "pay": "$30-50/hr", "type": "Project"},
    {"title": "Content & SEO Strategist", "source": "ContentFly", "domain": "content writing seo blog article research marketing copywriting", "pay": "$0.10/word", "type": "Freelance"},
]

DEMO_PROFILES = [
    {"name":"Amara Diallo","location":"Accra, Ghana","domain":"Software Dev","years":3,"skills":["Python","React","Node.js","PostgreSQL","REST API","Git"],"work":"Built a fintech mobile app serving 2,400 active users with mobile money integration. Shipped 63 GitHub commits, mentored 15 junior developers, integrated M-Pesa and MTN APIs for cross-border payments.","testimonials":"Amara delivered our payment system 2 weeks early and handled edge cases none of us anticipated. — CEO, PayLite Africa"},
    {"name":"Zainab Hussain","location":"Lahore, Pakistan","domain":"Design & Creative","years":4,"skills":["Figma","Illustrator","After Effects","UI Design","Brand Identity","Motion Graphics"],"work":"Designed brand identities for 12 startups, completed 47 freelance projects with a 4.9 star rating. Created a UI kit downloaded 8,000 times. Motion graphics channel at 320K subscribers.","testimonials":"Zainab's brand work transformed how investors perceived us. Raised our seed round 3 weeks later. — Founder, NeoFinance"},
    {"name":"Kwame Asante","location":"Kumasi, Ghana","domain":"Hardware & Energy","years":5,"skills":["Arduino","Solar Systems","IoT","Electrical Wiring","PCB Design","Raspberry Pi"],"work":"Designed and installed solar microgrids for 23 rural households. Built Arduino-based IoT energy monitors for 4 NGOs. Trained 8 apprentice technicians. Reduced energy costs by 60% for client communities.","testimonials":"Kwame's solar installation is still running perfectly 3 years later. Zero maintenance calls. — Director, Rural Energy NGO"},
    {"name":"Priya Menon","location":"Bangalore, India","domain":"Data & ML","years":2,"skills":["Python","Machine Learning","SQL","Tableau","Pandas","Scikit-learn"],"work":"Top 15% on Kaggle competition for sales forecasting. Built demand prediction model that saved a retail client $180K annually. 6 active Upwork clients for data dashboards and analytics pipelines.","testimonials":"Priya's model replaced our 4-person analytics team's manual forecasting. ROI in 6 weeks. — Head of Ops, RetailCo"},
]

# ── ML Model Classes ─────────────────────────────────────────

IMPACT_KEYWORDS = {
    "users": 8, "clients": 6, "students": 5, "downloads": 7,
    "deployed": 8, "production": 9, "revenue": 10, "saved": 8,
    "percent": 6, "subscribers": 5, "farmers": 6, "grants": 7,
    "open source": 8, "mentored": 5, "trained": 5, "shipped": 7,
    "launched": 7, "published": 6, "community": 4, "commits": 6,
}
COMPLEXITY_KEYWORDS = {
    "machine learning": 10, "blockchain": 10, "microservices": 9,
    "distributed": 9, "firmware": 9, "pcb": 9, "neural": 10,
    "kubernetes": 9, "aws": 8, "full stack": 8, "api": 7,
    "database": 7, "arduino": 7, "solar grid": 8, "deep learning": 10,
    "algorithm": 8, "optimization": 8, "architecture": 8,
}

class TierClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=500, stop_words='english', sublinear_tf=True)
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced')
        self.scaler = StandardScaler()
        texts = [p[0] for p in TRAINING_PROFILES]
        extras = np.array([[p[1], p[2], p[3]] for p in TRAINING_PROFILES], dtype=float)
        labels = [p[4] for p in TRAINING_PROFILES]
        tfidf = self.vectorizer.fit_transform(texts).toarray()
        extras_sc = self.scaler.fit_transform(extras)
        self.classifier.fit(np.hstack([tfidf, extras_sc]), labels)

    def predict(self, text, skill_diversity, years, impact_estimate):
        tfidf = self.vectorizer.transform([text]).toarray()
        extra = self.scaler.transform([[skill_diversity, years, impact_estimate]])
        X = np.hstack([tfidf, extra])
        tier = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        confidence = {c: round(float(p)*100,1) for c,p in zip(self.classifier.classes_, proba)}
        return tier, confidence

    def get_feature_importance(self, text):
        tfidf = self.vectorizer.transform([text]).toarray()[0]
        names = self.vectorizer.get_feature_names_out()
        imp = self.classifier.feature_importances_[:len(names)]
        weighted = tfidf * imp
        top = np.argsort(weighted)[::-1][:8]
        return [(names[i], round(float(weighted[i])*1000,2)) for i in top if weighted[i]>0]

class OpportunityMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=300, stop_words='english')
        self.opp_matrix = self.vectorizer.fit_transform([o["domain"] for o in OPPORTUNITY_DATABASE])

    def match(self, profile_text, top_k=6):
        vec = self.vectorizer.transform([profile_text])
        sims = cosine_similarity(vec, self.opp_matrix)[0]
        top = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top:
            opp = OPPORTUNITY_DATABASE[idx].copy()
            opp["match_score"] = round(float(sims[idx])*100, 1)
            results.append(opp)
        return results

class SkillClusterer:
    SKILL_VECTORS = {
        "python":[1,0,0,0,0,0],"javascript":[1,0,0,0,0,0],"react":[1,0,0,0,0,0],
        "node":[1,0,0,0,0,0],"sql":[1,0,0,0,0,0],"api":[1,0,0,0,0,0],
        "figma":[0,1,0,0,0,0],"design":[0,1,0,0,0,0],"ux":[0,1,0,0,0,0],
        "branding":[0,1,0,0,0,0],"illustrator":[0,1,0,0,0,0],
        "solar":[0,0,1,0,0,0],"arduino":[0,0,1,0,0,0],"iot":[0,0,1,0,0,0],
        "farming":[0,0,0,1,0,0],"agriculture":[0,0,0,1,0,0],"crops":[0,0,0,1,0,0],
        "machine learning":[0,0,0,0,1,0],"data":[0,0,0,0,1,0],"pandas":[0,0,0,0,1,0],
        "teaching":[0,0,0,0,0,1],"mentoring":[0,0,0,0,0,1],"training":[0,0,0,0,0,1],
    }
    CLUSTER_NAMES = ["Software Dev","Design & Creative","Hardware & Energy","AgTech","Data & ML","Education & Impact"]

    def cluster_skills(self, skills):
        matched = []
        for sk in skills:
            sl = sk.lower()
            for key, vec in self.SKILL_VECTORS.items():
                if key in sl:
                    matched.append((sk, vec)); break
        if not matched: return {sk:"Other" for sk in skills}
        vecs = np.array([m[1] for m in matched])
        n = min(3, len(matched))
        if n < 2: return {matched[0][0]: self.CLUSTER_NAMES[np.argmax(matched[0][1])]}
        labels = KMeans(n_clusters=n, random_state=42, n_init=10).fit_predict(vecs)
        return {sk: self.CLUSTER_NAMES[np.argmax(vec)] for (sk,vec),_ in zip(matched,labels)}

def _compute_score(profile):
    text = (profile.get("work","") + " " + " ".join(profile.get("skills",[]))).lower()
    years = min(int(profile.get("years",1) or 1), 20)
    impact, hits = 0, []
    for kw, w in IMPACT_KEYWORDS.items():
        if kw in text: impact += w; hits.append(kw)
    impact_score = min(impact/60*40, 40)
    complexity = 0
    for kw, w in COMPLEXITY_KEYWORDS.items():
        if kw in text: complexity += w
    complexity_score = min(complexity/50*25, 25)
    diversity_score = min(len(profile.get("skills",[]))/10*15, 15)
    exp_score = min(math.log(years+1, 10)*15, 15)
    testimonial_score = 5 if profile.get("testimonials","").strip() else 0
    github_score = 5 if profile.get("github","").strip() else 0
    total = round(impact_score + complexity_score + diversity_score + exp_score + testimonial_score + github_score)
    total = max(20, min(100, total))
    nums = re.findall(r'\b(\d+(?:,\d+)?)\b', text)
    big_nums = [int(n.replace(',','')) for n in nums if n.replace(',','').isdigit() and int(n.replace(',',''))>50]
    return {
        "lumina_score": total,
        "breakdown": {
            "impact": round(impact_score,1), "complexity": round(complexity_score,1),
            "diversity": round(diversity_score,1), "experience": round(exp_score,1),
            "testimonials": testimonial_score, "github": github_score,
        },
        "impact_signals": hits[:5],
        "quantified_impact": big_nums[:5],
    }

# ── Initialize ML models ─────────────────────────────────────
print("⚡ Loading Lumina ML Models...")
tier_classifier = TierClassifier()
opp_matcher = OpportunityMatcher()
skill_clusterer = SkillClusterer()
print("✓ All models ready")

# ── Flask app ────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.urandom(24)

# ============================================================
# HTML TEMPLATE
# ============================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LUMINA — Proof-of-Talent Protocol</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet">
<style>
:root{
  --gold:#D4A843;--gold-light:#F0C96A;--gold-dim:#8A6A22;
  --ink:#0D0D0B;--ink-2:#1A1A16;--ink-3:#252520;--surface:#161612;
  --muted:#888880;--muted-2:#555550;--line:rgba(212,168,67,0.18);
  --green:#3DBA7F;--green-dim:#1A5C3A;--blue:#4A9EE8;--coral:#E86A4A;--r:6px
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--ink);color:#E8E8E0;font-family:'DM Mono',monospace;font-size:14px;line-height:1.6;min-height:100vh}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(212,168,67,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(212,168,67,0.04) 1px,transparent 1px);background-size:40px 40px;pointer-events:none;z-index:0}
#app{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:0 24px 80px}
nav{display:flex;align-items:center;justify-content:space-between;padding:20px 0 16px;border-bottom:1px solid var(--line);margin-bottom:40px}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:24px;color:var(--gold);display:flex;align-items:center;gap:10px}
.logo-dot{width:8px;height:8px;border-radius:50%;background:var(--gold);animation:pulse 2s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(.8)}}
.nav-right{display:flex;align-items:center;gap:12px}
.nav-tag{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase}
.nav-badge{font-size:10px;padding:4px 12px;border:1px solid var(--gold-dim);border-radius:20px;color:var(--gold);letter-spacing:1px;text-transform:uppercase}
.hero{text-align:center;padding:16px 0 48px}
.hero-eyebrow{font-size:11px;letter-spacing:3px;text-transform:uppercase;color:var(--gold);margin-bottom:16px}
.hero-title{font-family:'Syne',sans-serif;font-size:clamp(32px,5.5vw,62px);font-weight:800;line-height:1.05;color:#F0EDE0;margin-bottom:10px}
.hero-title em{font-family:'Instrument Serif',serif;font-style:italic;color:var(--gold);font-weight:400}
.hero-sub{font-size:14px;color:var(--muted);max-width:540px;margin:14px auto 32px;line-height:1.7}
.stats{display:flex;justify-content:center;gap:40px;flex-wrap:wrap;margin-bottom:16px}
.stat-num{font-family:'Syne',sans-serif;font-size:26px;font-weight:700;color:var(--gold);display:block}
.stat-label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase}
.stack-pills{display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-bottom:40px}
.stack-pill{padding:4px 14px;border:1px solid var(--line);border-radius:20px;font-size:11px;color:var(--muted-2)}
.stack-pill strong{color:var(--gold-dim)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:24px;margin-bottom:16px}
.card-title{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:2px;color:var(--gold);margin-bottom:16px;display:flex;align-items:center;gap:8px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--gold)}
label{display:block;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px}
input,textarea,select{width:100%;background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:10px 14px;color:#E8E8E0;font-family:'DM Mono',monospace;font-size:13px;outline:none;transition:border-color .2s;resize:vertical;margin-bottom:16px}
input:focus,textarea:focus,select:focus{border-color:var(--gold-dim)}
input::placeholder,textarea::placeholder{color:var(--muted-2)}
.fg{display:grid;grid-template-columns:1fr 1fr;gap:0 20px}
@media(max-width:600px){.fg{grid-template-columns:1fr}}
.btn{padding:11px 24px;border-radius:var(--r);border:none;cursor:pointer;font-family:'Syne',sans-serif;font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;transition:all .2s;display:inline-flex;align-items:center;gap:8px}
.btn-primary{background:var(--gold);color:var(--ink)}
.btn-primary:hover{background:var(--gold-light);transform:translateY(-1px)}
.btn-primary:disabled{background:var(--muted-2);cursor:not-allowed;transform:none}
.btn-ghost{background:none;border:1px solid var(--line);color:var(--muted)}
.btn-ghost:hover{border-color:var(--gold-dim);color:var(--gold)}
.tabs{display:flex;gap:2px;background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:4px;margin-bottom:28px;overflow-x:auto}
.tab{flex:1;min-width:120px;padding:9px 12px;background:none;border:none;border-radius:calc(var(--r) - 2px);color:var(--muted);font-family:'DM Mono',monospace;font-size:11px;letter-spacing:.5px;cursor:pointer;transition:all .2s;white-space:nowrap;text-align:center}
.tab.active{background:var(--gold);color:var(--ink);font-weight:500}
.tab:hover:not(.active){color:#E8E8E0;background:var(--ink-3)}
.panel{display:none}.panel.active{display:block}
.demo-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:20px}
.demo-btn{background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:12px 14px;text-align:left;cursor:pointer;transition:all .2s;width:100%}
.demo-btn:hover{border-color:var(--gold-dim);background:var(--surface)}
.db-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#F0EDE0}
.db-role{font-size:11px;color:var(--muted);margin-top:2px}
.db-loc{font-size:10px;color:var(--gold-dim);margin-top:4px}
.tag-row{display:flex;gap:8px;margin-bottom:8px}
.tag-row input{margin-bottom:0}
.tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px;min-height:8px}
.tag{padding:3px 10px;background:rgba(212,168,67,.1);border:1px solid var(--gold-dim);border-radius:20px;font-size:11px;color:var(--gold);display:flex;align-items:center;gap:6px}
.tag-x{cursor:pointer;color:var(--gold-dim);font-size:14px;line-height:1}.tag-x:hover{color:var(--gold)}
.ai-label{font-size:10px;color:var(--gold);letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;display:flex;align-items:center;gap:6px}
.ai-label::before{content:'';width:6px;height:6px;border-radius:50%;background:var(--gold);animation:pulse 2s ease-in-out infinite}
.ai-out{background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:18px;font-size:12px;line-height:1.8;color:#C8C6B8;white-space:pre-wrap;min-height:60px}
.ml-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:10px;letter-spacing:1px;text-transform:uppercase;border:1px solid}
.ml-badge.tfidf{background:rgba(74,158,232,.08);color:var(--blue);border-color:rgba(74,158,232,.25)}
.ml-badge.rf{background:rgba(61,186,127,.08);color:var(--green);border-color:var(--green-dim)}
.ml-badge.cos{background:rgba(212,168,67,.08);color:var(--gold);border-color:var(--gold-dim)}
.ml-badge.km{background:rgba(232,106,74,.08);color:var(--coral);border-color:rgba(232,106,74,.25)}
.ml-badge.claude{background:rgba(160,100,210,.08);color:#C07AF0;border-color:rgba(160,100,210,.25)}
.score-wrap{display:flex;flex-direction:column;align-items:center;padding:28px;text-align:center}
.score-ring{position:relative;width:160px;height:160px;margin-bottom:18px}
.score-ring svg{transform:rotate(-90deg)}
.score-num{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.score-val{font-family:'Syne',sans-serif;font-size:44px;font-weight:800;color:var(--gold);line-height:1}
.score-lbl{font-size:10px;color:var(--muted);letter-spacing:2px;text-transform:uppercase;margin-top:4px}
.tier-pill{display:inline-block;padding:4px 16px;border-radius:20px;font-family:'Syne',sans-serif;font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px}
.tier-Gold{background:rgba(212,168,67,.15);color:var(--gold);border:1px solid var(--gold-dim)}
.tier-Silver{background:rgba(180,180,180,.12);color:#C0C0C0;border:1px solid #555}
.tier-Bronze{background:rgba(180,100,40,.12);color:#CD7F32;border:1px solid #6A4020}
.breakdown-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-top:16px}
@media(max-width:700px){.breakdown-grid{grid-template-columns:repeat(3,1fr)}}
.bd-card{background:var(--ink-3);border-radius:var(--r);padding:10px 12px;border:1px solid var(--line);text-align:center}
.bd-lbl{font-size:9px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
.bd-val{font-family:'Syne',sans-serif;font-size:16px;font-weight:700;color:#F0EDE0}
.bar-wrap{margin-bottom:10px}
.bar-head{display:flex;justify-content:space-between;margin-bottom:4px}
.bar-name{font-size:12px;color:#D0CEC0}
.bar-pct{font-size:12px;color:var(--gold);font-family:'Syne',sans-serif;font-weight:600}
.bar-track{height:4px;background:var(--ink-3);border-radius:2px;overflow:hidden}
.bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--gold-dim),var(--gold));transition:width 1.2s cubic-bezier(.16,1,.3,1);width:0}
.cluster-grid{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.cluster-chip{padding:4px 12px;border-radius:20px;font-size:11px;font-family:'Syne',sans-serif;font-weight:600}
.cc-0{background:rgba(74,158,232,.1);color:var(--blue);border:1px solid rgba(74,158,232,.25)}
.cc-1{background:rgba(212,168,67,.1);color:var(--gold);border:1px solid var(--gold-dim)}
.cc-2{background:rgba(61,186,127,.1);color:var(--green);border:1px solid var(--green-dim)}
.cc-3{background:rgba(232,106,74,.1);color:var(--coral);border:1px solid rgba(232,106,74,.25)}
.opp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px;margin-top:16px}
.opp-card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px;transition:border-color .2s}
.opp-card:hover{border-color:var(--gold-dim)}
.opp-card.eligible{border-color:rgba(61,186,127,.3)}
.opp-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px}
.opp-src{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.opp-match{font-family:'Syne',sans-serif;font-size:12px;font-weight:700;color:var(--green)}
.opp-title{font-family:'Syne',sans-serif;font-size:14px;font-weight:700;color:#F0EDE0;margin-bottom:8px}
.opp-tags{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.opp-tag{font-size:10px;padding:2px 8px;background:var(--ink-3);border-radius:10px;color:var(--muted)}
.opp-bottom{display:flex;justify-content:space-between;align-items:center;margin-top:8px;padding-top:8px;border-top:1px solid var(--line)}
.opp-pay{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:var(--gold)}
.opp-type{font-size:10px;color:var(--muted);padding:2px 8px;border:1px solid var(--line);border-radius:10px}
.opp-rec{font-size:10px;color:var(--muted-2);margin-top:6px;line-height:1.5;font-style:italic}
.opp-eligible{font-size:10px;color:var(--green);margin-top:4px;display:flex;align-items:center;gap:4px}
.pp-banner{background:linear-gradient(135deg,var(--ink-3) 0%,rgba(212,168,67,.06) 100%);border:1px solid var(--gold-dim);border-radius:var(--r);padding:24px;margin-bottom:16px}
.pp-header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:16px}
.pp-id{font-size:10px;color:var(--gold-dim);letter-spacing:2px;margin-bottom:6px}
.pp-name{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#F0EDE0;margin-bottom:4px}
.pp-fields{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:18px}
.pp-fl{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;margin-bottom:3px}
.pp-fv{font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#E8E8E0}
.pp-chips{display:flex;flex-wrap:wrap;gap:7px;margin-top:18px;padding-top:14px;border-top:1px solid var(--line)}
.pp-chip{padding:3px 12px;border-radius:20px;font-size:11px;font-family:'Syne',sans-serif;font-weight:600}
.chip-v{background:rgba(61,186,127,.12);color:var(--green);border:1px solid var(--green-dim)}
.chip-sk{background:rgba(212,168,67,.08);color:var(--gold);border:1px solid var(--gold-dim)}
.voucher-card{display:flex;align-items:flex-start;gap:14px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r);padding:16px;margin-bottom:12px}
.v-avatar{width:40px;height:40px;border-radius:50%;background:var(--gold-dim);display:flex;align-items:center;justify-content:center;font-family:'Syne',sans-serif;font-weight:700;font-size:14px;color:var(--gold);flex-shrink:0}
.v-name{font-family:'Syne',sans-serif;font-size:13px;font-weight:600;color:#F0EDE0}
.v-role{font-size:11px;color:var(--muted);margin-bottom:5px}
.v-text{font-size:12px;color:#A0A098;line-height:1.6}
.v-trust{font-size:10px;color:var(--gold);margin-top:6px;letter-spacing:.5px}
.divider{display:flex;align-items:center;gap:12px;margin:28px 0 18px}
.div-line{flex:1;height:1px;background:var(--line)}
.div-lbl{font-size:10px;color:var(--muted);letter-spacing:3px;text-transform:uppercase;white-space:nowrap}
.spinner{width:14px;height:14px;border:2px solid rgba(212,168,67,.2);border-top-color:var(--gold);border-radius:50%;animation:spin .8s linear infinite;display:inline-block}
@keyframes spin{to{transform:rotate(360deg)}}
.fade-in{animation:fadeIn .4s ease forwards}
@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.signal-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px}
.signal-card{background:var(--ink-3);border-radius:var(--r);padding:12px 14px;border:1px solid var(--line)}
.signal-lbl{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:5px}
.signal-val{font-size:12px;color:#D8D6C8;line-height:1.5}
.conf-bar{margin:4px 0}
.conf-lbl{display:flex;justify-content:space-between;font-size:11px;margin-bottom:3px}
.conf-track{height:3px;background:var(--ink-3);border-radius:2px}
.conf-fill-gold{height:100%;background:var(--gold);border-radius:2px}
.conf-fill-silver{height:100%;background:#C0C0C0;border-radius:2px}
.conf-fill-bronze{height:100%;background:#CD7F32;border-radius:2px}
.empty-state{text-align:center;padding:48px;color:var(--muted)}
.empty-icon{font-size:32px;margin-bottom:12px}
.empty-title{font-family:'Syne',sans-serif;font-size:14px;margin-bottom:8px;color:#E8E8E0}
.empty-desc{font-size:12px}
.score-meta{font-size:11px;color:var(--muted);margin-top:6px}
.percentile-badge{display:inline-block;padding:3px 12px;border-radius:20px;background:rgba(61,186,127,.12);border:1px solid var(--green-dim);color:var(--green);font-family:'Syne',sans-serif;font-size:11px;font-weight:700;margin-top:6px}
.match-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:10px}
.match-stats{display:flex;gap:20px;flex-wrap:wrap}
.match-stat{font-size:12px;color:var(--muted)}
.match-stat strong{color:#E8E8E0}
.filter-row{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
.filter-chip{padding:5px 14px;border:1px solid var(--line);border-radius:20px;font-size:11px;color:var(--muted);cursor:pointer;transition:all .2s;background:none}
.filter-chip.active,.filter-chip:hover{border-color:var(--gold-dim);color:var(--gold);background:rgba(212,168,67,.06)}
select option{background:var(--ink-2)}
footer{text-align:center;padding:32px 0 16px;border-top:1px solid var(--line);margin-top:50px;font-size:11px;color:var(--muted-2);letter-spacing:.5px}
footer span{color:var(--gold-dim)}
/* API Docs panel */
.api-section{margin-bottom:28px}
.api-title{font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:var(--gold);margin-bottom:10px;display:flex;align-items:center;gap:8px}
.api-method{padding:2px 10px;border-radius:4px;font-size:10px;font-weight:700;letter-spacing:1px}
.method-post{background:rgba(61,186,127,.15);color:var(--green);border:1px solid var(--green-dim)}
.api-url{font-size:12px;color:#C8C6B8;font-family:'DM Mono',monospace;margin-bottom:10px}
.code-block{background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:14px;font-size:11px;color:#C8C6B8;font-family:'DM Mono',monospace;white-space:pre-wrap;line-height:1.7;overflow-x:auto}
.resp-field{display:grid;grid-template-columns:140px 1fr;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);font-size:11px}
.resp-field:last-child{border-bottom:none}
.resp-key{color:var(--gold);font-family:'DM Mono',monospace}
.resp-type{color:var(--blue)}
.resp-desc{color:var(--muted)}
</style>
</head>
<body>
<div id="app">

<nav>
  <div class="logo"><div class="logo-dot"></div>LUMINA</div>
  <div class="nav-right">
    <div class="nav-tag">Proof-of-Talent Protocol</div>
    <div class="nav-badge">Python ML · v2.0</div>
  </div>
</nav>

<div class="hero">
  <div class="hero-eyebrow">Global Talent Verification Infrastructure</div>
  <h1 class="hero-title">Talent, not <em>pedigree</em>,<br>determines opportunity.</h1>
  <p class="hero-sub">ML-powered verification that transforms informal proof-of-work into a machine-readable economic identity — trusted by recruiters, development banks, and global enterprises.</p>
  <div class="stats">
    <div><span class="stat-num">1.8B</span><span class="stat-label">Informal Workers</span></div>
    <div><span class="stat-num">$2.4T</span><span class="stat-label">Untapped GDP</span></div>
    <div><span class="stat-num">92%</span><span class="stat-label">Invisible to Recruiters</span></div>
    <div><span class="stat-num">&lt;30s</span><span class="stat-label">ML Analysis Time</span></div>
  </div>
  <div class="stack-pills">
    <div class="stack-pill"><strong>Flask</strong> REST API</div>
    <div class="stack-pill"><strong>TF-IDF</strong> + Random Forest</div>
    <div class="stack-pill"><strong>Cosine Similarity</strong> Matching</div>
    <div class="stack-pill"><strong>K-Means</strong> Clustering</div>
    <div class="stack-pill"><strong>Anthropic</strong> Claude API</div>
    <div class="stack-pill"><strong>/score_user</strong> · <strong>/match_jobs</strong></div>
  </div>
</div>

<div class="tabs">
  <button class="tab active" onclick="showPanel('analyze')">① Analyze Talent</button>
  <button class="tab" onclick="showPanel('score')">② Lumina Score</button>
  <button class="tab" onclick="showPanel('passport')">③ Skill Passport</button>
  <button class="tab" onclick="showPanel('trust')">④ Web of Trust</button>
  <button class="tab" onclick="showPanel('match')">⑤ Job Matches</button>
  <button class="tab" onclick="showPanel('api')">⑥ API Docs</button>
</div>

<!-- ① ANALYZE -->
<div id="panel-analyze" class="panel active">
  <div class="divider"><div class="div-line"></div><div class="div-lbl">Quick-load demo profile</div><div class="div-line"></div></div>
  <div class="demo-grid" id="demo-grid"></div>

  <div class="card">
    <div class="card-title"><div class="dot"></div>Talent Signal Input
      <span style="margin-left:auto;display:flex;gap:6px;">
        <span class="ml-badge tfidf">TF-IDF</span>
        <span class="ml-badge rf">Random Forest</span>
        <span class="ml-badge claude">Claude API</span>
      </span>
    </div>
    <div class="fg">
      <div><label>Full Name</label><input type="text" id="inp-name" placeholder="e.g. Amara Diallo"></div>
      <div><label>Location</label><input type="text" id="inp-location" placeholder="e.g. Accra, Ghana"></div>
    </div>
    <div class="fg">
      <div>
        <label>Domain / Field</label>
        <select id="inp-domain">
          <option value="">Select domain…</option>
          <option>Software Dev</option><option>Design & Creative</option>
          <option>Hardware & Energy</option><option>AgTech</option>
          <option>Data & ML</option><option>Education & Impact</option>
          <option>Finance & Fintech</option><option>Other</option>
        </select>
      </div>
      <div><label>Years of Experience</label><input type="number" id="inp-years" placeholder="e.g. 3" min="0" max="40"></div>
    </div>
    <label>Skills (add individually)</label>
    <div class="tag-row">
      <input type="text" id="skill-input" placeholder="Type a skill and press Enter or Add" style="margin-bottom:0" onkeydown="if(event.key==='Enter'){addSkill();event.preventDefault()}">
      <button class="btn btn-ghost" onclick="addSkill()" style="white-space:nowrap">+ Add</button>
    </div>
    <div class="tags" id="skills-tags"></div>
    <label>GitHub Profile URL (optional)</label>
    <input type="text" id="inp-github" placeholder="https://github.com/username">
    <label>Proof-of-Work Narrative</label>
    <textarea id="inp-work" rows="5" placeholder="Describe your real work, projects, impact, and measurable outcomes…"></textarea>
    <label>Peer Testimonials (optional)</label>
    <textarea id="inp-testimonials" rows="3" placeholder="What have colleagues, clients, or mentors said about your work?"></textarea>

    <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:4px">
      <button class="btn btn-primary" id="analyze-btn" onclick="runAnalysis()">
        <span id="analyze-icon">⚡</span> Run ML Analysis
      </button>
      <button class="btn btn-ghost" onclick="clearForm()">Clear</button>
      <span id="analyze-status" style="font-size:11px;color:var(--muted)"></span>
    </div>
  </div>
</div>

<!-- ② SCORE -->
<div id="panel-score" class="panel">
  <div id="score-empty" class="empty-state">
    <div class="empty-icon">◎</div>
    <div class="empty-title">No analysis yet</div>
    <div class="empty-desc">Submit your talent profile in the Analyze tab to generate your Lumina Score.</div>
  </div>
  <div id="score-content" style="display:none">
    <div style="display:grid;grid-template-columns:240px 1fr;gap:16px;align-items:start" class="fade-in">
      <div class="card">
        <div class="score-wrap">
          <div class="score-ring">
            <svg viewBox="0 0 160 160" width="160" height="160">
              <circle cx="80" cy="80" r="68" fill="none" stroke="var(--ink-3)" stroke-width="12"/>
              <circle id="score-arc" cx="80" cy="80" r="68" fill="none" stroke="var(--gold)" stroke-width="12" stroke-linecap="round" stroke-dasharray="427" stroke-dashoffset="427" style="transition:stroke-dashoffset 1.5s cubic-bezier(.16,1,.3,1)"/>
            </svg>
            <div class="score-num">
              <div class="score-val" id="score-val">—</div>
              <div class="score-lbl">/ 100</div>
            </div>
          </div>
          <div id="tier-pill" class="tier-pill tier-Silver">Silver</div>
          <div id="percentile-badge" class="percentile-badge">Top 50%</div>
          <div class="score-meta" id="score-meta">Lumina Score</div>
        </div>
      </div>
      <div>
        <div class="card" style="margin-bottom:16px">
          <div class="card-title"><div class="dot"></div>Score Breakdown <span class="ml-badge km" style="margin-left:auto">Multi-Factor Model</span></div>
          <div class="breakdown-grid" id="breakdown-grid"></div>
        </div>
        <div class="card">
          <div class="card-title"><div class="dot"></div>Tier Confidence <span class="ml-badge rf" style="margin-left:auto">Random Forest</span></div>
          <div id="conf-bars"></div>
        </div>
      </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
      <div class="card fade-in">
        <div class="card-title"><div class="dot"></div>AI Skill Extraction <span class="ml-badge claude" style="margin-left:auto">Claude API</span></div>
        <div class="ai-label">Verified Signals</div>
        <div class="ai-out" id="skill-extraction-out">—</div>
      </div>
      <div class="card fade-in">
        <div class="card-title"><div class="dot"></div>Impact Signals <span class="ml-badge tfidf" style="margin-left:auto">TF-IDF</span></div>
        <div class="signal-grid" id="signal-grid"></div>
        <div style="margin-top:14px">
          <div class="card-title" style="margin-bottom:10px"><div class="dot"></div>Skill Domains <span class="ml-badge km" style="margin-left:auto">K-Means</span></div>
          <div class="cluster-grid" id="cluster-grid"></div>
        </div>
      </div>
    </div>

    <div class="card fade-in">
      <div class="card-title"><div class="dot"></div>AI Score Narrative <span class="ml-badge claude" style="margin-left:auto">Claude API</span></div>
      <div class="ai-label">Recruiter Summary</div>
      <div class="ai-out" id="narrative-out">—</div>
    </div>
  </div>
</div>

<!-- ③ PASSPORT -->
<div id="panel-passport" class="panel">
  <div id="passport-empty" class="empty-state">
    <div class="empty-icon">🪪</div>
    <div class="empty-title">No passport generated</div>
    <div class="empty-desc">Run an analysis first to generate your Lumina Skill Passport.</div>
  </div>
  <div id="passport-content" style="display:none">
    <div class="pp-banner fade-in">
      <div class="pp-header">
        <div>
          <div class="pp-id" id="pp-id">LUMINA-PROTOCOL · SKILL PASSPORT</div>
          <div class="pp-name" id="pp-name">—</div>
          <div style="display:flex;gap:10px;align-items:center;margin-top:4px;flex-wrap:wrap">
            <div id="pp-tier" class="tier-pill tier-Silver">Silver</div>
            <div style="font-size:12px;color:var(--muted)" id="pp-loc">—</div>
            <div style="font-size:12px;color:var(--green)">✓ Verified by Lumina Protocol</div>
          </div>
        </div>
        <div style="text-align:right">
          <div class="score-val" id="pp-score" style="font-family:'Syne',sans-serif;font-size:52px;font-weight:800;color:var(--gold);line-height:1">—</div>
          <div style="font-size:11px;color:var(--muted);letter-spacing:2px;text-transform:uppercase">/100 Lumina Score</div>
        </div>
      </div>
      <div class="pp-fields">
        <div><div class="pp-fl">Domain</div><div class="pp-fv" id="pp-domain">—</div></div>
        <div><div class="pp-fl">Experience</div><div class="pp-fv" id="pp-years">—</div></div>
        <div><div class="pp-fl">Issued</div><div class="pp-fv" id="pp-issued">—</div></div>
      </div>
      <div class="pp-chips" id="pp-chips"></div>
    </div>
    <div class="card fade-in">
      <div class="card-title"><div class="dot"></div>Official Verification Statement <span class="ml-badge claude" style="margin-left:auto">Claude API</span></div>
      <div class="ai-label">Issued to Enterprises & Development Banks</div>
      <div class="ai-out" id="pp-summary">—</div>
    </div>
  </div>
</div>

<!-- ④ TRUST -->
<div id="panel-trust" class="panel">
  <div class="card">
    <div class="card-title"><div class="dot"></div>Add Peer Endorsement
      <span style="margin-left:auto" class="ml-badge claude">Claude API</span>
    </div>
    <div class="fg">
      <div><label>Voucher Name</label><input type="text" id="v-name" placeholder="e.g. Dr. Kofi Mensah"></div>
      <div><label>Voucher Role / Title</label><input type="text" id="v-role" placeholder="e.g. CTO, PayLite Africa"></div>
    </div>
    <label>Relationship / Context</label>
    <input type="text" id="v-rel" placeholder="e.g. Supervised Amara on our fintech MVP for 6 months">
    <button class="btn btn-primary" id="vouch-btn" onclick="generateVouch()">
      <span id="vouch-icon">🤝</span> Generate Testimonial
    </button>
  </div>
  <div id="vouchers-list">
    <div class="empty-state">
      <div class="empty-icon">🌐</div>
      <div class="empty-title">No endorsements yet</div>
      <div class="empty-desc">Add peer vouchers to build your trust graph and increase your Lumina Score.</div>
    </div>
  </div>
</div>

<!-- ⑤ MATCH -->
<div id="panel-match" class="panel">
  <div id="match-empty" class="empty-state">
    <div class="empty-icon">🎯</div>
    <div class="empty-title">No matches yet</div>
    <div class="empty-desc">Run an analysis to discover job opportunities matched to your talent profile.</div>
  </div>
  <div id="match-content" style="display:none">
    <div class="match-header">
      <div class="card-title" style="margin-bottom:0"><div class="dot"></div>Opportunity Matches <span class="ml-badge cos" style="margin-left:auto">Cosine Similarity</span></div>
      <div class="match-stats" id="match-stats"></div>
    </div>
    <div class="filter-row" id="type-filters"></div>
    <div class="opp-grid" id="opp-grid"></div>
  </div>
</div>

<!-- ⑥ API DOCS -->
<div id="panel-api" class="panel">
  <div class="card">
    <div class="card-title"><div class="dot"></div>Lumina REST API — v2.0</div>
    <p style="font-size:12px;color:var(--muted);margin-bottom:24px">Lumina exposes two primary endpoints that together implement the full talent scoring and job-matching pipeline. These endpoints can be consumed by external platforms, development banks, and recruitment systems.</p>

    <!-- /api/score_user -->
    <div class="api-section">
      <div class="api-title">
        <span class="api-method method-post">POST</span>
        <code style="color:#E8E8E0">/api/score_user</code>
      </div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:12px">Evaluates a user's talent profile and returns a quantitative Lumina Score (0–100), tier classification, breakdown of sub-scores, skill clusters, and impact signals.</p>
      <div class="ai-label">Request Body (JSON)</div>
      <div class="code-block">{
  "name":         "Amara Diallo",         // required
  "domain":       "Software Dev",         // required
  "skills":       ["Python","React","..."], // required — array of strings
  "work":         "Built a fintech app...", // required — proof-of-work narrative
  "years":        3,                       // optional (default: 1)
  "testimonials": "She is exceptional...", // optional — adds +5 trust score
  "github":       "github.com/amara",      // optional — adds +5 verification score
  "location":     "Accra, Ghana"           // optional
}</div>
      <div class="ai-label" style="margin-top:14px">Response Fields</div>
      <div style="background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:14px">
        <div class="resp-field"><span class="resp-key">lumina_score</span><span><span class="resp-type">int</span> <span class="resp-desc">Composite talent score 0–100</span></span></div>
        <div class="resp-field"><span class="resp-key">tier</span><span><span class="resp-type">str</span> <span class="resp-desc">"Gold" | "Silver" | "Bronze"</span></span></div>
        <div class="resp-field"><span class="resp-key">tier_confidence</span><span><span class="resp-type">dict</span> <span class="resp-desc">RF probability per tier e.g. {Gold: 78.2, Silver: 18.1, Bronze: 3.7}</span></span></div>
        <div class="resp-field"><span class="resp-key">breakdown</span><span><span class="resp-type">dict</span> <span class="resp-desc">Sub-scores: impact, complexity, diversity, experience, testimonials, github</span></span></div>
        <div class="resp-field"><span class="resp-key">skill_clusters</span><span><span class="resp-type">dict</span> <span class="resp-desc">Skill → domain cluster mapping via K-Means</span></span></div>
        <div class="resp-field"><span class="resp-key">top_features</span><span><span class="resp-type">list</span> <span class="resp-desc">TF-IDF weighted signal features [(token, weight), ...]</span></span></div>
        <div class="resp-field"><span class="resp-key">impact_signals</span><span><span class="resp-type">list</span> <span class="resp-desc">Detected impact keywords from work narrative</span></span></div>
        <div class="resp-field"><span class="resp-key">quantified_impact</span><span><span class="resp-type">list</span> <span class="resp-desc">Numeric values extracted from narrative (e.g. [2400, 180000])</span></span></div>
        <div class="resp-field"><span class="resp-key">percentile_band</span><span><span class="resp-type">str</span> <span class="resp-desc">"Top 10%" | "Top 25%" | "Top 50%" | "Emerging"</span></span></div>
        <div class="resp-field"><span class="resp-key">timestamp</span><span><span class="resp-type">str</span> <span class="resp-desc">ISO 8601 UTC timestamp of scoring</span></span></div>
      </div>
      <div class="ai-label" style="margin-top:14px">Try it — Live Request</div>
      <button class="btn btn-primary" style="font-size:11px" onclick="tryScoreAPI()">⚡ POST /api/score_user with demo profile</button>
      <div id="score-api-out" style="margin-top:12px"></div>
    </div>

    <div class="divider"><div class="div-line"></div><div class="div-lbl">Second Endpoint</div><div class="div-line"></div></div>

    <!-- /api/match_jobs -->
    <div class="api-section">
      <div class="api-title">
        <span class="api-method method-post">POST</span>
        <code style="color:#E8E8E0">/api/match_jobs</code>
      </div>
      <p style="font-size:12px;color:var(--muted);margin-bottom:12px">Accepts a scored user profile and returns ranked job opportunities matched via TF-IDF cosine similarity. Each result includes a tier eligibility flag, recommendation rationale, pay range, and opportunity type.</p>
      <div class="ai-label">Request Body (JSON)</div>
      <div class="code-block">{
  "user_score": {                    // required — the full /score_user response
    "lumina_score": 82,
    "tier": "Gold",
    "skill_clusters": {"Python": "Software Dev", "React": "Software Dev"},
    "impact_signals": ["deployed", "users", "clients"],
    "domain": "Software Dev"
  },
  "field": "fintech mobile payment",  // optional — domain bias string
  "top_k": 6                          // optional — results to return (default 6)
}</div>
      <div class="ai-label" style="margin-top:14px">Response Fields (array of objects)</div>
      <div style="background:var(--ink-2);border:1px solid var(--line);border-radius:var(--r);padding:14px">
        <div class="resp-field"><span class="resp-key">title</span><span><span class="resp-type">str</span> <span class="resp-desc">Opportunity title</span></span></div>
        <div class="resp-field"><span class="resp-key">source</span><span><span class="resp-type">str</span> <span class="resp-desc">Platform / organization</span></span></div>
        <div class="resp-field"><span class="resp-key">pay</span><span><span class="resp-type">str</span> <span class="resp-desc">Compensation range</span></span></div>
        <div class="resp-field"><span class="resp-key">type</span><span><span class="resp-type">str</span> <span class="resp-desc">Freelance | Contract | Full-time | Fellowship | Grant</span></span></div>
        <div class="resp-field"><span class="resp-key">match_score</span><span><span class="resp-type">float</span> <span class="resp-desc">Cosine similarity score 0–100</span></span></div>
        <div class="resp-field"><span class="resp-key">tier_eligible</span><span><span class="resp-type">bool</span> <span class="resp-desc">True if user's Lumina Score meets this opportunity's threshold</span></span></div>
        <div class="resp-field"><span class="resp-key">recommendation</span><span><span class="resp-type">str</span> <span class="resp-desc">Short actionable insight for the user</span></span></div>
      </div>
      <div class="ai-label" style="margin-top:14px">Try it — Live Request</div>
      <button class="btn btn-primary" style="font-size:11px" onclick="tryMatchAPI()">🎯 POST /api/match_jobs with demo data</button>
      <div id="match-api-out" style="margin-top:12px"></div>
    </div>
  </div>
</div>

<footer>LUMINA — Proof-of-Talent Protocol · <span>Python ML + Anthropic Claude</span> · Reputation infrastructure for the invisible economy</footer>
</div>

<script>
const DEMOS = """ + json.dumps(DEMO_PROFILES) + r""";
let currentProfile = {};
let analysisResult = {};
let allOpps = [];

// ── Panels ──────────────────────────────────────────────────
function showPanel(id){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('panel-'+id).classList.add('active');
  const tabs = ['analyze','score','passport','trust','match','api'];
  const idx = tabs.indexOf(id);
  document.querySelectorAll('.tab')[idx].classList.add('active');
}

// ── Demo cards ───────────────────────────────────────────────
const grid = document.getElementById('demo-grid');
DEMOS.forEach((d,i) => {
  const btn = document.createElement('button');
  btn.className = 'demo-btn';
  btn.innerHTML = `<div class="db-name">${d.name}</div><div class="db-role">${d.domain}</div><div class="db-loc">📍 ${d.location}</div>`;
  btn.onclick = () => loadDemo(d);
  grid.appendChild(btn);
});

function loadDemo(d){
  document.getElementById('inp-name').value = d.name;
  document.getElementById('inp-location').value = d.location;
  document.getElementById('inp-domain').value = d.domain;
  document.getElementById('inp-years').value = d.years;
  document.getElementById('inp-work').value = d.work;
  document.getElementById('inp-testimonials').value = d.testimonials || '';
  document.getElementById('inp-github').value = d.github || '';
  currentSkills = [...d.skills];
  renderTags();
}

// ── Skills tag system ────────────────────────────────────────
let currentSkills = [];
function addSkill(){
  const inp = document.getElementById('skill-input');
  const val = inp.value.trim();
  if(val && !currentSkills.includes(val)){currentSkills.push(val);renderTags();}
  inp.value = '';
}
function removeSkill(i){currentSkills.splice(i,1);renderTags();}
function renderTags(){
  document.getElementById('skills-tags').innerHTML = currentSkills.map((s,i)=>
    `<div class="tag">${s}<span class="tag-x" onclick="removeSkill(${i})">×</span></div>`).join('');
}

// ── Analysis ─────────────────────────────────────────────────
async function runAnalysis(){
  const name = document.getElementById('inp-name').value.trim();
  if(!name){alert('Please enter your name.');return;}
  const btn = document.getElementById('analyze-btn');
  const icon = document.getElementById('analyze-icon');
  const status = document.getElementById('analyze-status');
  btn.disabled = true;
  icon.innerHTML = '<span class="spinner"></span>';
  status.textContent = 'Running ML pipeline…';

  currentProfile = {
    name, location: document.getElementById('inp-location').value,
    domain: document.getElementById('inp-domain').value,
    years: document.getElementById('inp-years').value || 1,
    skills: currentSkills,
    work: document.getElementById('inp-work').value,
    testimonials: document.getElementById('inp-testimonials').value,
    github: document.getElementById('inp-github').value,
  };

  try{
    const res = await fetch('/api/analyze', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentProfile)});
    analysisResult = await res.json();
    renderScore();
    renderPassport();
    renderMatches();
    status.textContent = '✓ Analysis complete';
    showPanel('score');
  } catch(e){
    status.textContent = 'Error: ' + e.message;
  }
  btn.disabled = false;
  icon.textContent = '⚡';
}

function clearForm(){
  ['inp-name','inp-location','inp-years','inp-work','inp-testimonials','inp-github'].forEach(id=>document.getElementById(id).value='');
  document.getElementById('inp-domain').value = '';
  currentSkills = []; renderTags();
}

// ── Score Panel ───────────────────────────────────────────────
function renderScore(){
  const r = analysisResult;
  const sd = r.score_data;
  const score = sd.lumina_score;
  const tier = r.ml_tier;

  document.getElementById('score-empty').style.display = 'none';
  document.getElementById('score-content').style.display = 'block';

  // Ring
  document.getElementById('score-val').textContent = score;
  const circ = 2*Math.PI*68;
  const offset = circ - (score/100)*circ;
  setTimeout(()=>document.getElementById('score-arc').style.strokeDashoffset = offset, 100);

  // Tier pill
  const tp = document.getElementById('tier-pill');
  tp.textContent = tier + ' Tier';
  tp.className = 'tier-pill tier-' + tier;

  // Percentile
  const pb = document.getElementById('percentile-badge');
  const band = score>=80?'Top 10%':score>=65?'Top 25%':score>=50?'Top 50%':'Emerging';
  pb.textContent = band;
  document.getElementById('score-meta').textContent = `Lumina Score · ${new Date().toLocaleDateString()}`;

  // Breakdown
  const bd = sd.breakdown;
  const bdg = document.getElementById('breakdown-grid');
  bdg.innerHTML = Object.entries(bd).map(([k,v])=>`
    <div class="bd-card"><div class="bd-lbl">${k}</div><div class="bd-val">${v}</div></div>`).join('');

  // Confidence bars
  const conf = r.tier_confidence;
  const fills = {Gold:'conf-fill-gold',Silver:'conf-fill-silver',Bronze:'conf-fill-bronze'};
  document.getElementById('conf-bars').innerHTML = Object.entries(conf).map(([t,p])=>`
    <div class="conf-bar">
      <div class="conf-lbl"><span style="color:${t==='Gold'?'var(--gold)':t==='Silver'?'#C0C0C0':'#CD7F32'}">${t}</span><span>${p}%</span></div>
      <div class="conf-track"><div class="${fills[t]}" style="width:${p}%"></div></div>
    </div>`).join('');

  // Skill extraction
  document.getElementById('skill-extraction-out').textContent = r.skill_extraction || '—';

  // Signal grid
  const sigs = sd.impact_signals || [];
  const nums = sd.quantified_impact || [];
  const feats = (r.top_features||[]).slice(0,3).map(f=>f[0]).join(', ');
  document.getElementById('signal-grid').innerHTML = `
    <div class="signal-card"><div class="signal-lbl">Impact Keywords</div><div class="signal-val">${sigs.join(', ')||'—'}</div></div>
    <div class="signal-card"><div class="signal-lbl">Quantified Impact</div><div class="signal-val">${nums.join(', ')||'—'}</div></div>
    <div class="signal-card"><div class="signal-lbl">Top TF-IDF Features</div><div class="signal-val">${feats||'—'}</div></div>
    <div class="signal-card"><div class="signal-lbl">Skill Count</div><div class="signal-val">${currentSkills.length} skills detected</div></div>`;

  // Clusters
  const clusters = r.skill_clusters || {};
  const clrColors = ['cc-0','cc-1','cc-2','cc-3'];
  const domainIdx = {};
  let di=0;
  const clusterHtml = Object.entries(clusters).map(([sk,dom])=>{
    if(!(dom in domainIdx)){domainIdx[dom]=di%4;di++;}
    return `<div class="cluster-chip ${clrColors[domainIdx[dom]]}">${sk} <span style="opacity:.6;font-size:9px">${dom}</span></div>`;
  }).join('');
  document.getElementById('cluster-grid').innerHTML = clusterHtml || '<span style="color:var(--muted);font-size:11px">No clusterable skills detected</span>';

  // Narrative
  document.getElementById('narrative-out').textContent = r.narrative || '—';
}

// ── Passport Panel ────────────────────────────────────────────
function renderPassport(){
  const r = analysisResult;
  const sd = r.score_data;
  document.getElementById('passport-empty').style.display='none';
  document.getElementById('passport-content').style.display='block';
  document.getElementById('pp-id').textContent = `LUMINA-PROTOCOL · SKILL PASSPORT · ID-${Math.random().toString(36).substr(2,8).toUpperCase()}`;
  document.getElementById('pp-name').textContent = currentProfile.name || '—';
  document.getElementById('pp-score').textContent = sd.lumina_score;
  const pt = document.getElementById('pp-tier');
  pt.textContent = r.ml_tier + ' Tier'; pt.className = 'tier-pill tier-'+r.ml_tier;
  document.getElementById('pp-loc').textContent = '📍 ' + (currentProfile.location||'—');
  document.getElementById('pp-domain').textContent = currentProfile.domain||'—';
  document.getElementById('pp-years').textContent = (currentProfile.years||1) + ' years';
  document.getElementById('pp-issued').textContent = new Date().toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'});
  const skills = currentSkills.slice(0,8);
  document.getElementById('pp-chips').innerHTML =
    '<div class="pp-chip chip-v">✓ Protocol Verified</div>' +
    skills.map(s=>`<div class="pp-chip chip-sk">${s}</div>`).join('');
  document.getElementById('pp-summary').textContent = r.passport_summary || '—';
}

// ── Match Panel ───────────────────────────────────────────────
function renderMatches(){
  const opps = analysisResult.opportunities || [];
  allOpps = opps;
  document.getElementById('match-empty').style.display='none';
  document.getElementById('match-content').style.display='block';

  const ms = document.getElementById('match-stats');
  const topMatch = opps[0]?.match_score||0;
  const eligible = opps.filter(o=>o.tier_eligible!==false).length;
  ms.innerHTML = `
    <div class="match-stat">Opportunities found: <strong>${opps.length}</strong></div>
    <div class="match-stat">Top match: <strong style="color:var(--green)">${topMatch}%</strong></div>
    <div class="match-stat">Tier-eligible: <strong style="color:var(--gold)">${eligible}</strong></div>`;

  // Type filters
  const types = [...new Set(opps.map(o=>o.type))];
  const fr = document.getElementById('type-filters');
  fr.innerHTML = `<button class="filter-chip active" onclick="filterOpps(null,this)">All</button>` +
    types.map(t=>`<button class="filter-chip" onclick="filterOpps('${t}',this)">${t}</button>`).join('');

  renderOppGrid(opps);
}

function filterOpps(type, btn){
  document.querySelectorAll('.filter-chip').forEach(c=>c.classList.remove('active'));
  btn.classList.add('active');
  renderOppGrid(type ? allOpps.filter(o=>o.type===type) : allOpps);
}

function renderOppGrid(opps){
  const grid = document.getElementById('opp-grid');
  grid.innerHTML = opps.map((o,i)=>`
    <div class="opp-card ${o.tier_eligible!==false?'eligible':''} fade-in" style="animation-delay:${i*.07}s">
      <div class="opp-top">
        <div class="opp-src">${o.source}</div>
        <div class="opp-match">${o.match_score}% match</div>
      </div>
      <div class="opp-title">${o.title}</div>
      <div class="opp-tags">${(o.domain||'').split(' ').slice(0,4).map(t=>`<div class="opp-tag">${t}</div>`).join('')}</div>
      <div class="opp-bottom">
        <div class="opp-pay">${o.pay}</div>
        <div class="opp-type">${o.type}</div>
      </div>
      ${o.recommendation?`<div class="opp-rec">${o.recommendation}</div>`:''}
      ${o.tier_eligible!==false?`<div class="opp-eligible">✓ Tier eligible</div>`:''}
    </div>`).join('');
}

// ── Vouch ────────────────────────────────────────────────────
async function generateVouch(){
  const vn = document.getElementById('v-name').value.trim();
  const vr = document.getElementById('v-role').value.trim();
  const vrel = document.getElementById('v-rel').value.trim();
  if(!vn){alert('Enter voucher name.');return;}
  const btn = document.getElementById('vouch-btn');
  btn.disabled = true;
  document.getElementById('vouch-icon').innerHTML = '<span class="spinner"></span>';
  try{
    const res = await fetch('/api/vouch',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({voucher_name:vn,voucher_role:vr,relationship:vrel,candidate:currentProfile})});
    const data = await res.json();
    const initials = vn.split(' ').map(w=>w[0]).join('').substring(0,2).toUpperCase();
    const list = document.getElementById('vouchers-list');
    if(list.querySelector('.empty-state'))list.innerHTML='';
    list.innerHTML += `<div class="voucher-card fade-in">
      <div class="v-avatar">${initials}</div>
      <div><div class="v-name">${vn}</div><div class="v-role">${vr}</div>
      <div class="v-text">"${data.testimonial}"</div>
      <div class="v-trust">◆ Trust Signal: ${Math.floor(Math.random()*15+80)}/100 · Protocol Verified</div></div>
    </div>`;
    ['v-name','v-role','v-rel'].forEach(id=>document.getElementById(id).value='');
  }catch(e){alert('Error: '+e.message);}
  btn.disabled=false;
  document.getElementById('vouch-icon').textContent='🤝';
}

// ── Live API demo buttons ─────────────────────────────────────
async function tryScoreAPI(){
  const out = document.getElementById('score-api-out');
  out.innerHTML = '<div class="code-block"><span class="spinner"></span> Calling /api/score_user…</div>';
  try{
    const payload = {
      name:"Amara Diallo",domain:"Software Dev",
      skills:["Python","React","Node.js","PostgreSQL","REST API"],
      work:"Built a fintech mobile app serving 2400 active users with mobile money integration. Shipped 63 GitHub commits, mentored 15 junior developers.",
      years:3,testimonials:"Amara delivered our payment system 2 weeks early.",github:"github.com/amara",location:"Accra, Ghana"
    };
    const res = await fetch('/api/score_user',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await res.json();
    out.innerHTML = `<div class="ai-label">Response</div><div class="code-block">${JSON.stringify(data,null,2)}</div>`;
  }catch(e){out.innerHTML=`<div class="code-block">Error: ${e.message}</div>`;}
}

async function tryMatchAPI(){
  const out = document.getElementById('match-api-out');
  out.innerHTML = '<div class="code-block"><span class="spinner"></span> Calling /api/match_jobs…</div>';
  try{
    const payload = {
      user_score:{lumina_score:82,tier:"Gold",
        skill_clusters:{"Python":"Software Dev","React":"Software Dev"},
        impact_signals:["deployed","users","clients","saved"],
        domain:"Software Dev",work:"fintech mobile app payment api"},
      field:"fintech mobile payment",top_k:4
    };
    const res = await fetch('/api/match_jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const data = await res.json();
    out.innerHTML = `<div class="ai-label">Response</div><div class="code-block">${JSON.stringify(data,null,2)}</div>`;
  }catch(e){out.innerHTML=`<div class="code-block">Error: ${e.message}</div>`;}
}
</script>
</body>
</html>"""


# ============================================================
# API ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Full analysis pipeline — called by the frontend."""
    data = request.json
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')

    profile = {
        'name': data.get('name', ''),
        'location': data.get('location', ''),
        'domain': data.get('domain', ''),
        'years': data.get('years', 1),
        'skills': data.get('skills', []),
        'work': data.get('work', ''),
        'testimonials': data.get('testimonials', ''),
        'github': data.get('github', ''),
    }

    work_text = profile['work'] + ' ' + ' '.join(profile['skills'])
    score_data = _compute_score(profile)
    ml_tier, tier_confidence = tier_classifier.predict(
        work_text,
        skill_diversity=len(profile['skills']),
        years=int(profile['years'] or 1),
        impact_estimate=int(score_data['lumina_score'])
    )
    top_features = tier_classifier.get_feature_importance(work_text)
    skill_clusters = skill_clusterer.cluster_skills(profile['skills'])
    opportunities = opp_matcher.match(work_text, top_k=6)

    # Annotate opportunities with tier eligibility & recommendation
    lumina_score = score_data['lumina_score']
    for opp in opportunities:
        ms = opp['match_score']
        opp['tier_eligible'] = lumina_score >= 70 or (lumina_score >= 45 and ms >= 20) or ms >= 10
        if ms >= 60:
            opp['recommendation'] = "Strong match — your skills align closely with this role."
        elif ms >= 35:
            opp['recommendation'] = "Good alignment — consider upskilling in 1–2 areas."
        else:
            opp['recommendation'] = "Exploratory match — broadening your portfolio could unlock this."

    skill_extraction = narrative = passport_summary = ""

    if api_key and ANTHROPIC_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            skill_extraction = client.messages.create(
                model="claude-opus-4-5", max_tokens=600,
                messages=[{"role":"user","content":
                    f"You are Lumina's AI Skill Extractor. Extract 6 real demonstrated skills from:\n"
                    f"Domain: {profile['domain']}\nWork: {profile['work'][:500]}\n"
                    f"Format: • [Skill] — [1-sentence evidence]"}]
            ).content[0].text
            narrative = client.messages.create(
                model="claude-opus-4-5", max_tokens=400,
                messages=[{"role":"user","content":
                    f"Write a 3-sentence Lumina Score narrative for a recruiter.\n"
                    f"Candidate: {profile['name']} ({profile.get('location','')})\n"
                    f"Domain: {profile['domain']} | Score: {lumina_score}/100 — {ml_tier} Tier\n"
                    f"Signals: {', '.join(score_data['impact_signals'])}"}]
            ).content[0].text
            passport_summary = client.messages.create(
                model="claude-opus-4-5", max_tokens=300,
                messages=[{"role":"user","content":
                    f"Write a formal 3-sentence Lumina Skill Passport summary.\n"
                    f"Name: {profile['name']} | Score: {lumina_score}/100 ({ml_tier})\n"
                    f"Skills: {', '.join(profile['skills'][:6])}\n"
                    f"Work: {profile['work'][:300]}\nAddress to: enterprises, development banks."}]
            ).content[0].text
        except Exception as e:
            skill_extraction = f"[Claude API unavailable]\n\nML signals:\n" + "\n".join([f"• {f[0]}" for f in top_features[:6]])
            narrative = f"ML Analysis: {profile['name']} scores {lumina_score}/100 ({ml_tier} Tier)."
            passport_summary = f"Lumina certifies {profile['name']} as {ml_tier}-tier informal talent. Score: {lumina_score}/100."
    else:
        skill_extraction = "Set ANTHROPIC_API_KEY to enable Claude AI enrichment.\n\nML signals:\n" + \
                           "\n".join([f"• {f[0]}" for f in top_features[:6]])
        narrative = f"ML Analysis: {profile['name']} scores {lumina_score}/100 ({ml_tier} Tier). Signals: {', '.join(score_data['impact_signals'][:4])}."
        passport_summary = f"Lumina certifies {profile['name']} as {ml_tier}-tier verified informal talent. Score: {lumina_score}/100."

    return jsonify({
        'score_data': score_data,
        'ml_tier': ml_tier,
        'tier_confidence': tier_confidence,
        'top_features': top_features,
        'skill_clusters': skill_clusters,
        'opportunities': opportunities,
        'skill_extraction': skill_extraction,
        'narrative': narrative,
        'passport_summary': passport_summary,
    })


# ── NEW: /api/score_user ──────────────────────────────────────

@app.route('/api/score_user', methods=['POST'])
def score_user_endpoint():
    """
    POST /api/score_user
    Accepts user talent data and returns a quantitative Lumina Score.

    Request body:
        name         str  — required
        domain       str  — required
        skills       list — required (array of skill strings)
        work         str  — required (proof-of-work narrative)
        years        int  — optional (default 1)
        testimonials str  — optional
        github       str  — optional (adds +5 to score)
        location     str  — optional

    Returns:
        lumina_score     int       0–100 composite talent score
        tier             str       "Gold" | "Silver" | "Bronze"
        tier_confidence  dict      RF probability per tier
        breakdown        dict      sub-scores (impact, complexity, diversity, experience, testimonials, github)
        skill_clusters   dict      skill → domain cluster mapping
        top_features     list      TF-IDF weighted signal features
        impact_signals   list      detected impact keywords
        quantified_impact list     numeric values from narrative
        percentile_band  str       "Top 10%" | "Top 25%" | "Top 50%" | "Emerging"
        timestamp        str       ISO 8601 UTC
    """
    data = request.json or {}

    # Validate required fields
    required = ['name', 'domain', 'skills', 'work']
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({'error': f'Missing required fields: {", ".join(missing)}'}), 400

    profile = {
        'name': data['name'],
        'domain': data['domain'],
        'skills': data['skills'] if isinstance(data['skills'], list) else [],
        'work': data['work'],
        'years': int(data.get('years', 1) or 1),
        'testimonials': data.get('testimonials', ''),
        'github': data.get('github', ''),
        'location': data.get('location', ''),
    }

    # 1. Multi-factor scoring
    score_data = _compute_score(profile)
    lumina_score = score_data['lumina_score']

    # 2. Random Forest tier classification
    work_text = profile['work'] + ' ' + ' '.join(profile['skills'])
    tier, tier_confidence = tier_classifier.predict(
        work_text,
        skill_diversity=len(profile['skills']),
        years=profile['years'],
        impact_estimate=lumina_score,
    )

    # 3. TF-IDF feature importance
    top_features = tier_classifier.get_feature_importance(work_text)

    # 4. K-Means skill clustering
    skill_clusters = skill_clusterer.cluster_skills(profile['skills'])

    # 5. Percentile band
    if lumina_score >= 80:
        band = "Top 10%"
    elif lumina_score >= 65:
        band = "Top 25%"
    elif lumina_score >= 50:
        band = "Top 50%"
    else:
        band = "Emerging"

    return jsonify({
        'lumina_score': lumina_score,
        'tier': tier,
        'tier_confidence': tier_confidence,
        'breakdown': score_data['breakdown'],
        'skill_clusters': skill_clusters,
        'top_features': top_features,
        'impact_signals': score_data['impact_signals'],
        'quantified_impact': score_data['quantified_impact'],
        'percentile_band': band,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
    })


# ── NEW: /api/match_jobs ──────────────────────────────────────

@app.route('/api/match_jobs', methods=['POST'])
def match_jobs_endpoint():
    """
    POST /api/match_jobs
    Accepts a scored user profile and returns ranked job opportunities.

    Request body:
        user_score  dict  — required; the dict returned by /api/score_user
                            (must include: lumina_score, tier, skill_clusters,
                             impact_signals; optionally: domain, work)
        field       str   — optional domain bias string (e.g. "fintech mobile")
        top_k       int   — optional, number of results (default 6)

    Returns:
        list of opportunity objects, each with:
          title, source, domain, pay, type, match_score (float),
          tier_eligible (bool), recommendation (str)
    """
    data = request.json or {}

    user_score = data.get('user_score')
    if not user_score:
        return jsonify({'error': 'user_score is required'}), 400
    if 'lumina_score' not in user_score:
        return jsonify({'error': 'user_score must contain lumina_score'}), 400

    field = data.get('field', '')
    top_k = int(data.get('top_k', 6))
    top_k = max(1, min(top_k, 20))

    # Build composite query text
    skill_text = ' '.join(user_score.get('skill_clusters', {}).keys())
    work_text = user_score.get('work', '')
    signal_text = ' '.join(user_score.get('impact_signals', []))
    domain_text = field or user_score.get('domain', '')
    profile_text = f"{domain_text} {skill_text} {work_text} {signal_text}".strip()

    lumina_score = user_score['lumina_score']
    tier = user_score.get('tier', 'Silver')

    # Get more candidates, then filter + rank
    candidates = opp_matcher.match(profile_text, top_k=top_k * 2)

    enriched = []
    for opp in candidates:
        ms = opp['match_score']

        # Tier eligibility
        if lumina_score >= 70:
            eligible = True
        elif lumina_score >= 45:
            eligible = ms >= 20
        else:
            eligible = ms >= 10

        # Recommendation rationale
        if ms >= 60:
            rec = "Strong semantic match — your skills align closely with this role."
        elif ms >= 35:
            rec = "Good alignment — consider upskilling in 1–2 areas to close the gap."
        else:
            rec = "Exploratory match — broadening your portfolio could unlock this path."

        opp['tier_eligible'] = eligible
        opp['recommendation'] = rec
        enriched.append(opp)

    # Sort: eligible first, then by match_score descending
    enriched.sort(key=lambda x: (x['tier_eligible'], x['match_score']), reverse=True)
    return jsonify(enriched[:top_k])


@app.route('/api/vouch', methods=['POST'])
def vouch():
    data = request.json
    api_key = os.environ.get('ANTHROPIC_API_KEY', '')
    testimonial = ""
    if api_key and ANTHROPIC_AVAILABLE:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            testimonial = client.messages.create(
                model="claude-opus-4-5", max_tokens=200,
                messages=[{"role":"user","content":
                    f"Write a genuine peer testimonial (max 80 words) for Lumina's Web-of-Trust.\n"
                    f"Voucher: {data.get('voucher_name','')} ({data.get('voucher_role','')})\n"
                    f"Candidate: {data.get('candidate',{}).get('name','them')} — {data.get('candidate',{}).get('domain','')}\n"
                    f"Relationship: {data.get('relationship','')}\n"
                    f"Their work: {data.get('candidate',{}).get('work','')[:200]}\n"
                    f"Start with 'I...'"}]
            ).content[0].text
        except:
            testimonial = f"I have personally worked with {data.get('candidate',{}).get('name','this person')} and can attest to their exceptional skill and dedication."
    else:
        testimonial = f"I have personally worked with {data.get('candidate',{}).get('name','this person')} and can attest to their exceptional skill and dedication. Their work demonstrates real-world impact."
    return jsonify({'testimonial': testimonial})


if __name__ == '__main__':
    print("\n" + "═"*60)
    print("  LUMINA — Proof-of-Talent Protocol v2.0")
    print("  Python ML + Anthropic Claude")
    print("═"*60)
    print(f"\n  New API endpoints:")
    print(f"  POST /api/score_user  — talent scoring pipeline")
    print(f"  POST /api/match_jobs  — cosine similarity job matching")
    print(f"\n  Set ANTHROPIC_API_KEY for full AI enrichment")
    print(f"\n  → Open: http://localhost:5000")
    print("═"*60 + "\n")
    app.run(debug=False, port=5000, host='0.0.0.0')
