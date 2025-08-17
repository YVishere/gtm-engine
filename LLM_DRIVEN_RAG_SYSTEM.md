# LLM-Driven RAG Email Generation System

## Overview

This system transforms the RAG (Retrieval-Augmented Generation) email engine from **80% hardcoded** logic to **80% LLM-driven** decision making, with intelligent GitHub API rate limiting.

## Key Improvements

### 1. **LLM Search Strategist** (`llm_search_strategist.py`)
- **Before**: Hardcoded search queries based on templates
- **After**: LLM decides what to search for, why, and how
- **Features**:
  - Analyzes opportunity context to generate strategic search queries
  - Considers API rate limits when planning searches
  - Provides reasoning for each search decision
  - Estimates API usage and tracks remaining requests

### 2. **Enhanced Action Transparency** (`enhanced_action_transparency.py`)
- **Before**: Simple template-based reasoning messages
- **After**: Comprehensive tracking of LLM decisions with outcomes
- **Features**:
  - Records every LLM decision with context and confidence
  - Tracks API usage against configurable limits
  - Provides real-time rate limit status updates
  - Generates session analytics and recommendations

### 3. **LLM Outcome Assessor** (`llm_outcome_assessor.py`)
- **Before**: Basic success/failure metrics
- **After**: Intelligent assessment with adaptive learning
- **Features**:
  - LLM evaluates search strategy effectiveness
  - Provides gap analysis and improvement recommendations
  - Learns from successful patterns for future optimization
  - Generates adaptive next-step recommendations

## GitHub API Rate Limiting

### Configuration (`config.py`)
```python
GITHUB_MAX_REQUESTS_PER_SESSION = 15  # Conservative limit
GITHUB_MAX_REQUESTS_PER_HOUR = 100    # Hourly cap
GITHUB_REQUEST_DELAY = 1              # Seconds between requests
```

### Smart Usage Strategy
1. **LLM Awareness**: Every LLM prompt includes remaining API budget
2. **Strategic Planning**: LLM prioritizes high-value searches
3. **Real-time Tracking**: Continuous monitoring of usage vs limits
4. **Adaptive Behavior**: System adjusts strategy based on remaining quota

## LLM Decision Points

### 1. Search Strategy Generation
```python
# LLM Context Provided:
{
    "opportunity_context": {
        "original_question": "...",
        "extracted_technologies": ["react", "jwt"],
        "problem_type": "implementation_guidance",
        "technical_complexity": 6
    },
    "api_constraints": {
        "requests_remaining": 12,
        "max_total_requests": 15,
        "warning": "Be strategic - GitHub API requests are limited!"
    }
}

# LLM Decision Output:
{
    "search_queries": [
        {
            "query": "react jwt refresh token authentication",
            "priority": "high",
            "reasoning": "Directly matches user's specific need",
            "expected_repos": 8
        }
    ],
    "risk_assessment": {
        "api_usage_estimate": 3
    }
}
```

### 2. Repository Analysis Strategy
```python
# LLM Context:
{
    "discovered_repositories": [...],
    "api_constraints": {
        "requests_remaining": 8,
        "analysis_budget": 6,
        "warning": "Only 8 API requests remaining!"
    }
}

# LLM Decision:
{
    "target_repositories": ["facebook/react", "auth0/node-jsonwebtoken"],
    "analysis_depth": "medium",
    "reasoning": "These repos have highest stars and match tech stack",
    "estimated_api_usage": 4
}
```

### 3. Outcome Assessment
```python
# LLM evaluates results and provides:
{
    "overall_success_score": 0.85,
    "strengths": ["Found high-quality repositories", "Efficient API usage"],
    "recommendations": ["Search for error handling patterns"],
    "lessons_learned": ["React + JWT queries work better with 'hooks' keyword"],
    "next_actions": ["Analyze top 2 repositories in detail"]
}
```

## System Flow

1. **Purpose Detection** (LLM-driven)
   - Analyzes opportunity context
   - Determines optimal search strategy
   - Sets success criteria

2. **Strategic Search** (LLM-planned)
   - Generates targeted search queries
   - Considers API budget constraints
   - Prioritizes by expected value

3. **Repository Analysis** (LLM-guided)
   - Selects most promising repositories
   - Determines analysis depth
   - Focuses on relevant patterns

4. **Outcome Assessment** (LLM-evaluated)
   - Evaluates strategy effectiveness
   - Identifies gaps and improvements
   - Provides adaptive recommendations

5. **Learning Loop** (LLM-enhanced)
   - Records successful patterns
   - Updates strategy preferences
   - Improves future decisions

## Rate Limiting Implementation

### Real-time Monitoring
```python
class APIRateLimit:
    max_requests_per_session: int = 15
    requests_used: int = 0
    
    def can_make_request(self, estimated_requests: int = 1) -> bool:
        remaining = self.max_requests_per_session - self.requests_used
        return remaining >= estimated_requests
```

### LLM-Aware Planning
- Every LLM prompt includes current API status
- LLM plans searches within available budget
- System stops gracefully when limits approached
- Provides fallback strategies for low-quota scenarios

### Usage Analytics
```python
{
    "total_api_requests_used": 12,
    "max_requests_allowed": 15,
    "remaining_requests": 3,
    "usage_percentage": 80.0,
    "average_confidence": 0.85,
    "requests_per_minute": 2.1
}
```

## Benefits

1. **Intelligent Decision Making**: LLM considers context, constraints, and goals
2. **API Efficiency**: Smart planning prevents quota exhaustion
3. **Adaptive Learning**: System improves from successful patterns
4. **Full Transparency**: Every decision is logged with reasoning
5. **Quality Focus**: Prioritizes high-value results over quantity

## Testing

Run the test script to see the system in action:

```bash
python test_llm_rag_system.py
```

This will demonstrate:
- LLM-driven search strategy generation
- Real-time API rate limit tracking
- Repository analysis with LLM guidance
- Comprehensive outcome assessment
- Session analytics and recommendations

## Configuration Variables

Key settings in `config.py`:

```python
# API Rate Limiting
GITHUB_MAX_REQUESTS_PER_SESSION = 15  # Adjust based on your needs
GITHUB_MAX_REQUESTS_PER_HOUR = 100    # GitHub API limit

# LLM Decision Making
LLM_CONFIDENCE_THRESHOLD = 0.7        # Minimum confidence for decisions
MAX_SEARCH_QUERIES_PER_OPPORTUNITY = 4
MAX_REPOSITORIES_TO_ANALYZE = 5
```

## Result

The system now makes **maximum use of LLM intelligence** while **respecting API constraints**, providing a much more sophisticated and adaptive approach to RAG email generation.
