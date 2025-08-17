"""LLM Processor Analysis and Integration utilities for RAG system."""

import logging
from typing import Dict, Any, Optional
from llm_processor import LLMProcessor, LegacyLLMProcessor


class LLMProcessorAnalyzer:
    """Analyze and provide proper access to LLM calling capabilities."""
    
    def __init__(self, llm_processor: LLMProcessor):
        self.llm_processor = llm_processor
        self.logger = logging.getLogger(self.__class__.__name__)
        self.capabilities = self._analyze_capabilities()
        
    def _analyze_capabilities(self) -> Dict[str, Any]:
        """Analyze the LLM processor's available methods and capabilities."""
        
        processor_type = type(self.llm_processor.processor).__name__
        available_methods = [method for method in dir(self.llm_processor.processor) 
                           if not method.startswith('__')]
        
        capabilities = {
            'processor_type': processor_type,
            'available_methods': available_methods,
            'has_direct_ollama': hasattr(self.llm_processor.processor, '_call_ollama'),
            'has_process_single': hasattr(self.llm_processor.processor, '_process_single_content'),
            'has_wrapper_explanation': hasattr(self.llm_processor, 'generate_opportunity_explanation'),
            'recommended_approach': self._determine_best_approach(processor_type)
        }
        
        self.logger.info(f"Analyzed {processor_type} - Recommended approach: {capabilities['recommended_approach']}")
        return capabilities
    
    def _determine_best_approach(self, processor_type: str) -> str:
        """Determine the best approach for calling LLM based on processor type."""
        
        if processor_type == 'LegacyLLMProcessor':
            return 'direct_ollama'
        elif processor_type == 'VectorizedLLMProcessor':
            # VectorizedLLMProcessor uses internal components that create LegacyLLMProcessor
            # We should use the same pattern as SummaryGenerator and ContentProcessor
            return 'vectorized_internal'
        else:
            return 'wrapper_method'
    
    def call_llm_properly(self, prompt: str) -> str:
        """Call LLM using the most appropriate method for the current processor."""
        
        try:
            approach = self.capabilities['recommended_approach']
            
            if approach == 'direct_ollama' and self.capabilities['has_direct_ollama']:
                # Direct call to LegacyLLMProcessor
                return self.llm_processor.processor._call_ollama(prompt)
                
            elif approach == 'vectorized_internal':
                # VectorizedLLMProcessor - use the same pattern as its internal components
                # Create LegacyLLMProcessor like SummaryGenerator does
                from llm_processor import LegacyLLMProcessor
                internal_processor = LegacyLLMProcessor()
                return internal_processor._call_ollama(prompt)
                
            elif approach == 'wrapper_method':
                # Use processor wrapper method if available
                # This would need adaptation for specific prompt types
                self.logger.warning("Wrapper method approach not fully implemented")
                return ""
                
            else:
                self.logger.error(f"No suitable LLM calling method found for {self.capabilities['processor_type']}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to call LLM: {e}")
            return ""
    
    def parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response using the appropriate parser."""
        
        try:
            if self.capabilities['has_direct_ollama']:
                return self.llm_processor.processor._parse_llm_response(response)
            else:
                # Use legacy processor for parsing
                legacy_processor = LegacyLLMProcessor()
                return legacy_processor._parse_llm_response(response)
                
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return {}


class RAGLLMIntegration:
    """Proper LLM integration for RAG purpose detection."""
    
    def __init__(self, llm_processor: LLMProcessor):
        self.analyzer = LLMProcessorAnalyzer(llm_processor)
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def generate_purpose_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Generate RAG search purpose using proper LLM integration."""
        
        self.logger.info(f"Calling LLM via {self.analyzer.capabilities['recommended_approach']} approach")
        
        # Make the LLM call using the proper method
        response = self.analyzer.call_llm_properly(prompt)
        
        if not response:
            self.logger.warning("LLM returned empty response")
            return {}
        
        # Parse the response
        parsed_result = self.analyzer.parse_llm_response(response)
        
        if not parsed_result:
            self.logger.warning("Failed to parse LLM response")
            return {}
            
        self.logger.info("Successfully generated purpose via LLM")
        return parsed_result
    
    def generate_email_with_llm(self, prompt: str) -> str:
        """Generate email content using proper LLM integration."""
        
        self.logger.info(f"Generating email via {self.analyzer.capabilities['recommended_approach']} approach")
        
        # Make the LLM call using the proper method
        response = self.analyzer.call_llm_properly(prompt)
        
        if not response:
            self.logger.warning("LLM returned empty response for email generation")
            return ""
            
        # For email generation, we expect plain text response, not JSON
        self.logger.info("Successfully generated email content via LLM")
        return response.strip()
    
    def test_llm_integration(self) -> bool:
        """Test if LLM integration is working properly."""
        
        test_prompt = """Test prompt for LLM integration. Return JSON: {"status": "working", "message": "LLM integration successful"}"""
        
        try:
            result = self.generate_purpose_with_llm(test_prompt)
            return bool(result and result.get('status') == 'working')
        except Exception as e:
            self.logger.error(f"LLM integration test failed: {e}")
            return False
