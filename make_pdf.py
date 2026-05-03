import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,11))
text = """
Main Topic: Multilingual PDF Agent.

This document discusses the development and evaluation of an advanced multilingual PDF constrained conversational agent.

Key Findings:
1. The agent effectively processes text from PDFs.
2. It can translate queries across 40+ languages.
3. Hallucinations are heavily reduced.

The purpose of this document is to serve as a testing artifact for the retrieval system.
The key conclusions are that the system works perfectly and is ready for production.
"""
fig.text(0.1, 0.5, text, wrap=True)
fig.savefig('sample.pdf')
