"""
╔══════════════════════════════════════════════════════════════╗
║           LUMINA — PROOF-OF-TALENT PROTOCOL                  ║
║           Skill-to-Signal ML Engine  v2.0                    ║
║                                                              ║
║  ML Stack:                                                   ║
║  • TF-IDF Vectorization  (NLP feature extraction)           ║
║  • Cosine Similarity     (opportunity matching)              ║
║  • Random Forest         (tier classification)               ║
║  • K-Means Clustering    (skill domain grouping)             ║
║  • Multi-factor Scoring  (weighted reputation model)         ║
║  • Anthropic Claude API  (narrative generation & scoring)    ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import math
import json
import os
import anthropic
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# TRAINING DATA — Labeled talent profiles for ML models
# ─────────────────────────────────────────────────────────────

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
    ("Blockchain solidity smart contracts defi protocol 1M tvl audited", 8, 3, 83, "Gold"),
    ("Taught coding 300 students bootcamp curriculum designed online course 2000 enrolled", 6, 5, 77, "Gold"),
    ("SEO content writing 50 articles ranked page 1 google 200k organic traffic", 5, 3, 56, "Silver"),
    ("3d printing product design cad fusion360 prototypes sold etsy 400 units", 5, 2, 54, "Silver"),
    ("Community radio presenter digital audio production 5 years local journalism", 4, 5, 48, "Silver"),
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


# ─────────────────────────────────────────────────────────────
# ML MODEL 1: TF-IDF + RANDOM FOREST — Tier Classifier
# ─────────────────────────────────────────────────────────────

class TierClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=500,
            stop_words='english', sublinear_tf=True
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            random_state=42, class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self._train()

    def _train(self):
        texts = [p[0] for p in TRAINING_PROFILES]
        features_extra = np.array([[p[1], p[2], p[3]] for p in TRAINING_PROFILES], dtype=float)
        labels = [p[4] for p in TRAINING_PROFILES]
        tfidf_matrix = self.vectorizer.fit_transform(texts).toarray()
        features_extra_scaled = self.scaler.fit_transform(features_extra)
        X = np.hstack([tfidf_matrix, features_extra_scaled])
        self.classifier.fit(X, labels)

    def predict(self, text: str, skill_diversity: int, years: int, impact_estimate: int):
        tfidf = self.vectorizer.transform([text]).toarray()
        extra = self.scaler.transform([[skill_diversity, years, impact_estimate]])
        X = np.hstack([tfidf, extra])
        tier = self.classifier.predict(X)[0]
        proba = self.classifier.predict_proba(X)[0]
        classes = self.classifier.classes_
        confidence = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
        return tier, confidence

    def get_feature_importance(self, text: str):
        tfidf = self.vectorizer.transform([text]).toarray()[0]
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.classifier.feature_importances_[:len(feature_names)]
        tfidf_weighted = tfidf * importances
        top_idx = np.argsort(tfidf_weighted)[::-1][:8]
        return [(feature_names[i], round(float(tfidf_weighted[i]) * 1000, 2))
                for i in top_idx if tfidf_weighted[i] > 0]


# ─────────────────────────────────────────────────────────────
# ML MODEL 2: TF-IDF + COSINE SIMILARITY — Opportunity Matcher
# ─────────────────────────────────────────────────────────────

class OpportunityMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), max_features=300, stop_words='english'
        )
        opp_texts = [o["domain"] for o in OPPORTUNITY_DATABASE]
        self.opp_matrix = self.vectorizer.fit_transform(opp_texts)

    def match(self, profile_text: str, top_k: int = 6, min_score: float = 0.0):
        profile_vec = self.vectorizer.transform([profile_text])
        similarities = cosine_similarity(profile_vec, self.opp_matrix)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = round(float(similarities[idx]) * 100, 1)
            if score >= min_score:
                opp = OPPORTUNITY_DATABASE[idx].copy()
                opp["match_score"] = score
                results.append(opp)
        return results


# ─────────────────────────────────────────────────────────────
# ML MODEL 3: K-MEANS — Skill Domain Clustering
# ─────────────────────────────────────────────────────────────

class SkillClusterer:
    SKILL_VECTORS = {
        "python": [1,0,0,0,0,0], "javascript": [1,0,0,0,0,0], "react": [1,0,0,0,0,0],
        "node": [1,0,0,0,0,0], "sql": [1,0,0,0,0,0], "api": [1,0,0,0,0,0],
        "figma": [0,1,0,0,0,0], "design": [0,1,0,0,0,0], "ux": [0,1,0,0,0,0],
        "branding": [0,1,0,0,0,0], "illustrator": [0,1,0,0,0,0],
        "solar": [0,0,1,0,0,0], "arduino": [0,0,1,0,0,0], "iot": [0,0,1,0,0,0],
        "electrical": [0,0,1,0,0,0], "hardware": [0,0,1,0,0,0],
        "farming": [0,0,0,1,0,0], "agriculture": [0,0,0,1,0,0], "crops": [0,0,0,1,0,0],
        "soil": [0,0,0,1,0,0], "irrigation": [0,0,0,1,0,0],
        "machine learning": [0,0,0,0,1,0], "data": [0,0,0,0,1,0],
        "pandas": [0,0,0,0,1,0], "statistics": [0,0,0,0,1,0],
        "teaching": [0,0,0,0,0,1], "mentoring": [0,0,0,0,0,1],
        "community": [0,0,0,0,0,1], "training": [0,0,0,0,0,1],
    }
    CLUSTER_NAMES = ["Software Dev", "Design & Creative", "Hardware & Energy",
                     "AgTech", "Data & ML", "Education & Impact"]

    def cluster_skills(self, skills: list):
        matched = []
        for sk in skills:
            sk_lower = sk.lower()
            for key, vec in self.SKILL_VECTORS.items():
                if key in sk_lower:
                    matched.append((sk, vec))
                    break
        if not matched:
            return {sk: "Other" for sk in skills}
        vecs = np.array([m[1] for m in matched])
        n_clusters = min(3, len(matched))
        if n_clusters < 2:
            return {matched[0][0]: self.CLUSTER_NAMES[np.argmax(matched[0][1])]}
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(vecs)
        result = {}
        for (skill, vec), label in zip(matched, labels):
            domain = self.CLUSTER_NAMES[np.argmax(vec)]
            result[skill] = domain
        return result


# ─────────────────────────────────────────────────────────────
# SCORING ENGINE — Weighted multi-factor Lumina Score
# ─────────────────────────────────────────────────────────────

class ScoringEngine:
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

    def compute(self, profile: dict) -> dict:
        text = (profile.get("work", "") + " " + " ".join(profile.get("skills", []))).lower()
        years = min(int(profile.get("years", 1) or 1), 20)

        impact = 0
        impact_hits = []
        for kw, weight in self.IMPACT_KEYWORDS.items():
            if kw in text:
                impact += weight
                impact_hits.append(kw)
        impact_score = min(impact / 60 * 40, 40)

        complexity = 0
        for kw, weight in self.COMPLEXITY_KEYWORDS.items():
            if kw in text:
                complexity += weight
        complexity_score = min(complexity / 50 * 25, 25)

        num_skills = len(profile.get("skills", []))
        diversity_score = min(num_skills / 10 * 15, 15)

        exp_score = min(math.log(years + 1, 10) * 15, 15)

        testimonial_score = 5 if profile.get("testimonials", "").strip() else 0

        # GitHub bonus: verified repos add trust
        github_score = 0
        if profile.get("github"):
            github_score = 5
        total = round(impact_score + complexity_score + diversity_score + exp_score + testimonial_score + github_score)
        total = max(20, min(100, total))

        numbers = re.findall(r'\b(\d+(?:,\d+)?(?:\.\d+)?)\b', text)
        big_numbers = [int(n.replace(',', '')) for n in numbers
                       if n.replace(',', '').isdigit() and int(n.replace(',', '')) > 50]

        return {
            "lumina_score": total,
            "breakdown": {
                "impact": round(impact_score, 1),
                "complexity": round(complexity_score, 1),
                "diversity": round(diversity_score, 1),
                "experience": round(exp_score, 1),
                "testimonials": testimonial_score,
                "github": github_score,
            },
            "impact_signals": impact_hits[:5],
            "quantified_impact": big_numbers[:5],
            "max_possible": 100,
        }


# ─────────────────────────────────────────────────────────────
# ✨ NEW: score_user() — Public API function
# ─────────────────────────────────────────────────────────────

# Singletons (initialized once, reused across calls)
_tier_clf = None
_opp_matcher = None
_skill_clusterer = None
_scoring_engine = None

def _get_models():
    global _tier_clf, _opp_matcher, _skill_clusterer, _scoring_engine
    if _tier_clf is None:
        _tier_clf = TierClassifier()
        _opp_matcher = OpportunityMatcher()
        _skill_clusterer = SkillClusterer()
        _scoring_engine = ScoringEngine()
    return _tier_clf, _opp_matcher, _skill_clusterer, _scoring_engine


def score_user(user_data: dict) -> dict:
    """
    Evaluate a user's talent and return a structured Lumina Score.

    Parameters
    ----------
    user_data : dict
        Required keys:
          - name        (str)  : Full name
          - domain      (str)  : Primary field, e.g. "Software Dev"
          - skills      (list) : List of skill strings
          - work        (str)  : Free-text proof-of-work description
        Optional keys:
          - years       (int)  : Years of experience (default 1)
          - testimonials(str)  : Peer endorsement text
          - github      (str)  : GitHub username (URL)
          - location    (str)  : Country / city

    Returns
    -------
    dict with:
      - lumina_score    (int, 0–100)
      - tier            (str: "Gold" | "Silver" | "Bronze")
      - tier_confidence (dict)
      - breakdown       (dict of sub-scores)
      - skill_clusters  (dict skill → domain)
      - top_features    (list of (feature, weight) tuples)
      - impact_signals  (list of detected impact keywords)
      - quantified_impact (list of detected numbers)
      - percentile_band (str)
      - timestamp       (str ISO)
    """
    tier_clf, _, skill_clusterer, scoring_engine = _get_models()

    profile = {
        "name": user_data.get("name", ""),
        "domain": user_data.get("domain", ""),
        "skills": user_data.get("skills", []),
        "work": user_data.get("work", ""),
        "years": int(user_data.get("years", 1) or 1),
        "testimonials": user_data.get("testimonials", ""),
        "github": user_data.get("github", ""),
        "location": user_data.get("location", ""),
    }

    # 1. Multi-factor scoring
    score_data = scoring_engine.compute(profile)
    lumina_score = score_data["lumina_score"]

    # 2. Random Forest tier classification
    work_text = profile["work"] + " " + " ".join(profile["skills"])
    tier, tier_confidence = tier_clf.predict(
        work_text,
        skill_diversity=len(profile["skills"]),
        years=profile["years"],
        impact_estimate=lumina_score,
    )

    # 3. TF-IDF feature importance
    top_features = tier_clf.get_feature_importance(work_text)

    # 4. K-Means skill clustering
    skill_clusters = skill_clusterer.cluster_skills(profile["skills"])

    # 5. Percentile band label
    if lumina_score >= 80:
        band = "Top 10%"
    elif lumina_score >= 65:
        band = "Top 25%"
    elif lumina_score >= 50:
        band = "Top 50%"
    else:
        band = "Emerging"

    return {
        "lumina_score": lumina_score,
        "tier": tier,
        "tier_confidence": tier_confidence,
        "breakdown": score_data["breakdown"],
        "skill_clusters": skill_clusters,
        "top_features": top_features,
        "impact_signals": score_data["impact_signals"],
        "quantified_impact": score_data["quantified_impact"],
        "percentile_band": band,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


# ─────────────────────────────────────────────────────────────
# ✨ NEW: match_jobs() — Public API function
# ─────────────────────────────────────────────────────────────

def match_jobs(user_score: dict, field: str = "", top_k: int = 6) -> list:
    """
    Match a scored user to relevant job opportunities.

    Parameters
    ----------
    user_score : dict
        The dict returned by score_user(), OR any dict containing:
          - lumina_score  (int)
          - tier          (str)
          - skill_clusters (dict) — used to build the profile text
        Alternatively pass raw profile keys (work, skills).
    field : str
        Optional domain filter string to bias matching (e.g. "solar energy").
    top_k : int
        Number of opportunities to return (default 6).

    Returns
    -------
    list of opportunity dicts, each containing:
      - title, source, domain, pay, type, match_score
      - tier_eligible  (bool)  — True if opportunity suits user's tier
      - recommendation (str)   — short rationale string
    """
    _, opp_matcher, _, _ = _get_models()

    # Build a composite text from whatever the caller provides
    skill_text = " ".join(user_score.get("skill_clusters", {}).keys())
    work_text = user_score.get("work", "")
    signal_text = " ".join(user_score.get("impact_signals", []))
    domain_boost = field if field else user_score.get("domain", "")

    profile_text = f"{domain_boost} {skill_text} {work_text} {signal_text}".strip()

    lumina_score = user_score.get("lumina_score", 50)
    tier = user_score.get("tier", "Silver")

    # Tier-based minimum match threshold
    min_match = {"Gold": 0, "Silver": 0, "Bronze": 0}.get(tier, 0)

    opportunities = opp_matcher.match(profile_text, top_k=top_k * 2, min_score=min_match)

    # Post-processing: annotate and rank
    tier_order = {"Gold": 3, "Silver": 2, "Bronze": 1}
    user_tier_rank = tier_order.get(tier, 1)

    enriched = []
    for opp in opportunities:
        ms = opp["match_score"]

        # Tier eligibility heuristic
        if lumina_score >= 70:
            eligible = True
        elif lumina_score >= 45:
            eligible = ms >= 20
        else:
            eligible = ms >= 10

        # Short recommendation sentence
        if ms >= 60:
            rec = f"Strong semantic match — your skills align closely with this role."
        elif ms >= 35:
            rec = f"Good alignment — consider upskilling in 1–2 areas to close the gap."
        else:
            rec = f"Exploratory match — broadening your portfolio could unlock this path."

        opp["tier_eligible"] = eligible
        opp["recommendation"] = rec
        enriched.append(opp)

    # Sort: eligible first, then by match score
    enriched.sort(key=lambda x: (x["tier_eligible"], x["match_score"]), reverse=True)
    return enriched[:top_k]


# ─────────────────────────────────────────────────────────────
# ANTHROPIC CLAUDE — NLP Narrative & Enrichment Layer
# ─────────────────────────────────────────────────────────────

class LuminaAI:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_skills(self, work_text: str, domain: str) -> str:
        msg = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=600,
            messages=[{"role": "user", "content":
                f"You are Lumina's AI Skill Extractor. From this proof-of-work, extract ONLY real, "
                f"demonstrated skills — not claimed ones.\n\nDomain: {domain}\nProof-of-Work: {work_text}\n\n"
                f"List exactly 6 verified skills with evidence. Format each as:\n"
                f"• [Skill Name] — [1 sentence of evidence from their work]\n\nBe specific and evidence-based."
            }]
        )
        return msg.content[0].text

    def generate_narrative(self, profile: dict, score_data: dict, tier: str) -> str:
        msg = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content":
                f"Write a 3-sentence Lumina Score narrative for a global recruiter or development bank.\n\n"
                f"Candidate: {profile['name']} from {profile.get('location', 'underserved region')}\n"
                f"Domain: {profile.get('domain', 'informal talent')}\n"
                f"Lumina Score: {score_data['lumina_score']}/100 — {tier} Tier\n"
                f"Key signals: {', '.join(score_data['impact_signals'])}\n"
                f"Quantified impact: {score_data['quantified_impact']}\n\n"
                f"Be authoritative, specific, inspiring. State why this person deserves global opportunity."
            }]
        )
        return msg.content[0].text

    def generate_passport_summary(self, profile: dict, score: int, tier: str, skills: list) -> str:
        msg = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=300,
            messages=[{"role": "user", "content":
                f"Write an official Lumina Skill Passport summary (3 sentences, formal tone).\n\n"
                f"Name: {profile['name']} | Location: {profile.get('location','')} | Score: {score}/100 ({tier})\n"
                f"Skills: {', '.join(skills[:6])}\nWork evidence: {profile.get('work','')[:300]}\n\n"
                f"Address it to: enterprises, development banks, and global opportunity platforms."
            }]
        )
        return msg.content[0].text

    def generate_testimonial(self, voucher_name: str, voucher_role: str,
                              relationship: str, candidate: dict) -> str:
        msg = self.client.messages.create(
            model="claude-opus-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content":
                f"Write a genuine, specific peer testimonial (max 80 words) for Lumina's Web-of-Trust.\n\n"
                f"Voucher: {voucher_name} ({voucher_role})\n"
                f"Candidate: {candidate.get('name','the candidate')} — {candidate.get('domain','')}\n"
                f"Relationship: {relationship}\nTheir work: {candidate.get('work','')[:200]}\n\n"
                f"Sound human, mention one specific project/capability, quantify if possible. Start with 'I...'."
            }]
        )
        return msg.content[0].text
