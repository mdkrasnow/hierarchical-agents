# Scoring Rubric Guide for Multi-Agent Review System

## Overview

This guide explains how to use the scoring rubric to evaluate answers from hierarchical agents. The rubric focuses on **information coverage, detail, and writing style** while explicitly de-emphasizing factual correctness.

**Key Principle: Rate how well information is presented, not whether it's factually accurate.**

## Scoring System

- **Total Score Range**: 0-100 points
- **Five Quality Tiers**: Excellent (90-100), Good (75-89), Adequate (60-74), Poor (40-59), Inadequate (0-39)
- **Weighted Dimensions**: Five dimensions with different importance weights

## Scoring Dimensions

### 1. Information Coverage (30% weight) 🎯
**Most Important Dimension**

Evaluates how completely the answer addresses the question or task.

**What to Look For:**
- Does it address all parts of the question?
- Are the main aspects covered?
- Does it consider edge cases or nuances?
- Is anything important missing?

**Examples:**

**Excellent (90-100 points):**
> Question: "How do I set up a home office?"
> 
> Answer covers: workspace location, furniture needs, lighting requirements, technology setup, ergonomics, noise management, organization systems, legal/tax considerations, and maintenance routines.

**Poor (40-59 points):**
> Question: "How do I set up a home office?"
> 
> Answer covers: desk and chair selection only, missing technology, lighting, organization, ergonomics, and other key aspects.

### 2. Detail & Specificity (25% weight) 🔍
**Second Most Important**

Evaluates the depth and specificity of information provided.

**What to Look For:**
- Are there specific examples and concrete details?
- Is terminology precise?
- Is information actionable?
- Are there concrete numbers, steps, or evidence?

**Examples:**

**Excellent (90-100 points):**
> "Use a desk height of 28-30 inches, ensuring your elbows are at 90-degree angles when typing. Position your monitor 20-26 inches away with the top of the screen at eye level. Install a adjustable LED desk lamp providing 1000-3000 lux illumination."

**Poor (40-59 points):**
> "Get a good desk and make sure the lighting is adequate for your work."

### 3. Structure & Coherence (20% weight) 📋

Evaluates logical organization and clarity of presentation.

**What to Look For:**
- Is there a clear logical flow?
- Are ideas well-organized?
- Are transitions smooth?
- Is it easy to follow?

**Examples:**

**Excellent (90-100 points):**
```
# Home Office Setup Guide

## 1. Planning Your Space
- Assess available rooms
- Consider noise levels
- Evaluate natural light

## 2. Essential Furniture
- Desk selection criteria
- Chair ergonomics
- Storage solutions

## 3. Technology Requirements
[etc.]
```

**Poor (40-59 points):**
> Random mix of furniture advice, then technology, then back to furniture, then lighting, with no clear organization or transitions between topics.

### 4. Style & Tone (15% weight) ✍️

Evaluates writing quality, professionalism, and readability.

**What to Look For:**
- Is the writing clear and engaging?
- Is the tone appropriate for the audience?
- Is it professional yet accessible?
- Are there grammar/style issues?

**Examples:**

**Excellent (90-100 points):**
> "Creating an effective home office requires careful attention to both physical comfort and productivity factors. Let's explore the key elements that will transform your space into a professional work environment."

**Poor (40-59 points):**
> "idk just get some stuff for ur desk and make sure its ok i guess. lighting is important too but whatever works."

### 5. Instruction Following (10% weight) ✅
**Lowest Priority**

Evaluates adherence to specific instructions and constraints.

**What to Look For:**
- Did it follow format requirements?
- Were word limits respected?
- Were specific constraints addressed?
- Did it answer what was actually asked?

**Examples:**

**Excellent (90-100 points):**
> Question asks for "5 bullet points under 200 words total"
> Answer provides exactly 5 bullet points, 180 words total

**Poor (40-59 points):**
> Question asks for "5 bullet points under 200 words total"
> Answer provides 3 paragraphs, 450 words total

## Scoring Process

### Step 1: Read the Answer Completely
- Understand the full response before scoring
- Note the original question/task context

### Step 2: Evaluate Each Dimension
- Score each dimension independently (0-100 scale)
- Use the tier descriptions as guidelines
- Consider the specific question context

### Step 3: Calculate Final Score
```
Final Score = (Coverage × 0.30) + (Detail × 0.25) + (Structure × 0.20) + (Style × 0.15) + (Instructions × 0.10)
```

### Step 4: Round and Validate
- Round to nearest integer
- Ensure score is between 0-100
- Sanity check against overall impression

## Common Scoring Scenarios

### High Coverage + Low Detail (Score: ~65-75)
- Touches on all relevant topics
- But provides only surface-level information
- Common in rushed or overly broad answers

### High Detail + Poor Coverage (Score: ~55-70)
- Very detailed in specific areas
- But misses important aspects of the question
- Common when answering only part of a multi-part question

### Good Content + Poor Structure (Score: ~70-80)
- Contains good information
- But poorly organized or hard to follow
- Can significantly impact usability despite good content

### Perfect Instructions + Weak Content (Score: ~45-65)
- Follows format perfectly
- But content is shallow or incomplete
- Low weight on instructions prevents high scores

## Important Guidelines

### ❌ What NOT to Penalize For:
- **Factual inaccuracies** (explicitly de-emphasized)
- **Disagreement with your opinions**
- **Different approaches or methodologies**
- **Creative or unconventional solutions**

### ✅ What TO Evaluate:
- **How comprehensively the answer addresses the question**
- **How much useful detail is provided**
- **How clearly and logically information is presented**
- **How professional and readable the writing is**
- **How well instructions are followed**

### 🎯 Focus Areas by Question Type:

**Technical Questions:**
- Emphasize detail and specificity
- Look for concrete steps and examples
- Structure becomes very important

**Conceptual Questions:**
- Emphasize coverage and coherence
- Look for comprehensive exploration of ideas
- Style and clarity become more important

**Creative Questions:**
- Emphasize coverage of creative possibilities
- Look for detailed examples and specifics
- Style and engagement become more important

## Quality Benchmarks

### Score 90-100: Exceptional
- Comprehensive, detailed, well-structured, engaging, perfectly follows instructions
- Could serve as a reference example
- No significant gaps or issues

### Score 75-89: Strong
- Good coverage with solid detail, well-organized, professional
- Minor gaps or areas for improvement
- Clearly helpful and well-executed

### Score 60-74: Adequate
- Covers basics with reasonable detail, generally organized
- Some gaps in coverage or detail, minor issues with structure/style
- Functional but not exceptional

### Score 40-59: Weak
- Incomplete coverage, limited detail, organization issues
- Missing important aspects, difficult to follow
- Needs significant improvement

### Score 0-39: Poor
- Severely incomplete, very limited detail, major issues
- Fails to adequately address the question
- Not useful in current form

## Example Scoring Walkthrough

**Question:** "Explain how to prepare for a job interview."

**Sample Answer Analysis:**

**Coverage Assessment:**
- Covers: resume review, company research, question preparation, outfit selection
- Missing: follow-up strategy, salary negotiation prep, references
- **Score: 75** (good coverage of main areas, minor gaps)

**Detail Assessment:**
- Specific examples of research sources
- Concrete question types with examples
- Detailed outfit guidelines
- **Score: 80** (good specificity, actionable advice)

**Structure Assessment:**
- Clear sections with logical flow
- Good transitions between topics
- Easy to follow chronologically
- **Score: 85** (well-organized, minor room for improvement)

**Style Assessment:**
- Professional tone throughout
- Engaging and encouraging
- Clear, readable language
- **Score: 78** (professional and clear, could be more engaging)

**Instructions Assessment:**
- Followed format requirements
- Appropriate length
- Answered the question asked
- **Score: 90** (excellent adherence)

**Final Calculation:**
(75 × 0.30) + (80 × 0.25) + (85 × 0.20) + (78 × 0.15) + (90 × 0.10)
= 22.5 + 20 + 17 + 11.7 + 9
= **80.2 → 80 points**

This represents a **Good** answer that covers the main points well with solid detail and organization.

## Troubleshooting Common Issues

### "The answer seems factually wrong"
- **Don't penalize for this directly**
- Focus on how well the (possibly incorrect) information is presented
- Consider if poor coverage might be due to factual errors leading to missing important aspects

### "I disagree with the approach"
- **Don't penalize for different approaches**
- Evaluate how well their chosen approach is explained and detailed
- Focus on presentation quality, not approach preference

### "The answer is too long/short"
- **Only penalize if specific length requirements were given**
- Otherwise, evaluate based on whether the length serves the content well
- Long answers aren't automatically better if they're repetitive or unfocused

### "I'm between two tier scores"
- **Use the specific score ranges within tiers**
- Consider which direction the specific strengths/weaknesses point
- When truly uncertain, round down for consistency

## Quality Assurance

### Before Finalizing Scores:
1. **Reread the original question** - Am I evaluating the right thing?
2. **Check dimension weights** - Am I appropriately weighting coverage and detail?
3. **Verify calculation** - Does my math add up correctly?
4. **Sanity check** - Does the final score match my overall impression?
5. **Review guidelines** - Am I following the de-emphasis on factual correctness?

### Red Flags for Re-evaluation:
- Score doesn't match your gut feeling about answer quality
- Heavy penalty for factual issues (should be minimal)
- Inconsistent scoring across similar dimensions
- Not considering the specific question context

---

*This rubric prioritizes information presentation quality over factual accuracy, enabling fair evaluation of AI-generated content based on structure, completeness, and communication effectiveness.*