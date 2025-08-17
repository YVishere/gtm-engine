"""Test script for LLM-driven RAG email generation system."""

import sys
import os
import logging
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.getcwd())

from config import Config
from enhanced_rag_email_engine import EnhancedRAGEmailEngine
from models import ProcessedContent, ScrapedContent, ContentSource

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_test_opportunity():
    """Create a test opportunity for authentication guidance"""
    
    original_content = ScrapedContent(
        title="How to implement JWT refresh tokens in React with error handling?",
        content="""I'm building a React application and need to implement JWT refresh tokens 
        with proper error handling for authentication. The app needs to handle token expiration 
        gracefully and redirect users to login when necessary. I'm looking for production-ready 
        examples that show how to structure the authentication context, handle API interceptors, 
        and manage token storage securely. Any recommendations for libraries or patterns that 
        work well with React hooks?""",
        url="https://stackoverflow.com/questions/test",
        timestamp=datetime.now(),
        author="test_developer",
        source=ContentSource.STACKOVERFLOW,
        tags=['react', 'jwt', 'authentication', 'refresh-tokens'],
        upvotes=25,
        language="javascript"
    )
    
    return ProcessedContent(
        original=original_content,
        summary="Developer needs JWT refresh token implementation in React with error handling",
        key_topics=['react', 'jwt', 'refresh-tokens', 'authentication', 'error-handling'],
        relevance_score=0.9,
        urgency_level='medium',
        business_opportunity="High-value authentication implementation opportunity",
        gtm_explanation="Developer implementing authentication presents strong sales opportunity"
    )

def test_llm_driven_rag_system():
    """Test the LLM-driven RAG system with rate limiting"""
    
    print("🧪 TESTING LLM-DRIVEN RAG EMAIL SYSTEM")
    print("="*50)
    
    # Check for GitHub API key
    github_token = Config.GITHUB_API_KEY
    if not github_token:
        print("❌ ERROR: GITHUB_API_KEY not found in environment variables")
        print("Please set GITHUB_API_KEY in your .env file")
        return
    
    print(f"✅ GitHub API key found")
    print(f"📊 API Rate Limits:")
    print(f"   • Max requests per session: {Config.GITHUB_MAX_REQUESTS_PER_SESSION}")
    print(f"   • Max requests per hour: {Config.GITHUB_MAX_REQUESTS_PER_HOUR}")
    print(f"   • LLM confidence threshold: {Config.LLM_CONFIDENCE_THRESHOLD}")
    
    try:
        # Initialize the enhanced RAG engine
        rag_engine = EnhancedRAGEmailEngine(github_token)
        print(f"✅ LLM-driven RAG engine initialized successfully")
        
        # Create test opportunity
        test_opportunity = create_test_opportunity()
        print(f"✅ Test opportunity created: '{test_opportunity.original.title[:50]}...'")
        
        # Test the full LLM-driven pipeline
        print(f"\n🚀 Starting LLM-driven email generation...")
        rag_engine.generate_rag_email_solutions([test_opportunity])
        
        # Show final analytics
        analytics = rag_engine.action_tracker.generate_session_analytics()
        
        print(f"\n📊 FINAL SESSION ANALYTICS:")
        print(f"="*50)
        
        session_summary = analytics['session_summary']
        print(f"🔢 API Usage:")
        print(f"   • Total requests used: {session_summary['total_api_requests_used']}")
        print(f"   • Requests remaining: {session_summary['remaining_requests']}")
        print(f"   • Usage percentage: {session_summary['usage_percentage']:.1f}%")
        
        print(f"\n🧠 LLM Decisions:")
        print(f"   • Total decisions made: {session_summary['llm_decisions_made']}")
        print(f"   • Average confidence: {session_summary['average_confidence']:.2f}")
        print(f"   • Search executions: {session_summary['search_executions']}")
        print(f"   • Repository analyses: {session_summary['repository_analyses']}")
        
        llm_analysis = analytics['llm_decision_analysis']
        print(f"\n📈 Decision Quality:")
        print(f"   • Average success score: {llm_analysis['average_success_score']:.2f}")
        print(f"   • API estimation accuracy: {llm_analysis['api_estimation_accuracy']:.2f}")
        
        if analytics['recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in analytics['recommendations'][:3]:
                print(f"   • {rec}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📧 Check the emails/ directory for generated email solutions")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_llm_driven_rag_system()
