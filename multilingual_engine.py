"""
Multilingual Legal AI Engine
Supports Hindi, Tamil, Telugu, Bengali, Marathi, and other Indian languages
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageDetection:
    """Language detection result"""
    language_code: str
    language_name: str
    confidence: float
    script: str

@dataclass
class TranslationResult:
    """Translation result with metadata"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    translation_method: str

class MultilingualLegalEngine:
    """Multilingual support for Indian legal queries"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.gemini_model = None
        
        if gemini_api_key and gemini_api_key != "AIzaSyDGlQJJhJJhJJhJJhJJhJJhJJhJJhJJhJJ":
            import google.generativeai as genai
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        # Supported languages
        self.supported_languages = {
            "hi": {"name": "Hindi", "script": "Devanagari", "native": "हिंदी"},
            "ta": {"name": "Tamil", "script": "Tamil", "native": "தமிழ்"},
            "te": {"name": "Telugu", "script": "Telugu", "native": "తెలుగు"},
            "bn": {"name": "Bengali", "script": "Bengali", "native": "বাংলা"},
            "mr": {"name": "Marathi", "script": "Devanagari", "native": "मराठी"},
            "gu": {"name": "Gujarati", "script": "Gujarati", "native": "ગુજરાતી"},
            "kn": {"name": "Kannada", "script": "Kannada", "native": "ಕನ್ನಡ"},
            "ml": {"name": "Malayalam", "script": "Malayalam", "native": "മലയാളം"},
            "pa": {"name": "Punjabi", "script": "Gurmukhi", "native": "ਪੰਜਾਬੀ"},
            "or": {"name": "Odia", "script": "Odia", "native": "ଓଡ଼ିଆ"},
            "as": {"name": "Assamese", "script": "Bengali", "native": "অসমীয়া"},
            "en": {"name": "English", "script": "Latin", "native": "English"}
        }
        
        # Legal terminology translations
        self.legal_terms = self._load_legal_terminology()
        
        # Language patterns for detection
        self.language_patterns = self._initialize_language_patterns()
    
    def _load_legal_terminology(self) -> Dict[str, Dict[str, str]]:
        """Load legal terminology translations"""
        return {
            "contract": {
                "hi": "अनुबंध",
                "ta": "ஒப்பந்தம்",
                "te": "ఒప్పందం",
                "bn": "চুক্তি",
                "mr": "करार",
                "gu": "કરાર",
                "kn": "ಒಪ್ಪಂದ",
                "ml": "കരാർ",
                "pa": "ਇਕਰਾਰ",
                "en": "contract"
            },
            "law": {
                "hi": "कानून",
                "ta": "சட்டம்",
                "te": "చట్టం",
                "bn": "আইন",
                "mr": "कायदा",
                "gu": "કાયદો",
                "kn": "ಕಾನೂನು",
                "ml": "നിയമം",
                "pa": "ਕਾਨੂੰਨ",
                "en": "law"
            },
            "court": {
                "hi": "न्यायालय",
                "ta": "நீதிமன்றம்",
                "te": "న్యాయస్థానం",
                "bn": "আদালত",
                "mr": "न्यायालय",
                "gu": "અદાલત",
                "kn": "ನ್ಯಾಯಾಲಯ",
                "ml": "കോടതി",
                "pa": "ਅਦਾਲਤ",
                "en": "court"
            },
            "section": {
                "hi": "धारा",
                "ta": "பிரிவு",
                "te": "సెక్షన్",
                "bn": "ধারা",
                "mr": "कलम",
                "gu": "કલમ",
                "kn": "ವಿಭಾಗ",
                "ml": "വകുപ്പ്",
                "pa": "ਧਾਰਾ",
                "en": "section"
            },
            "bail": {
                "hi": "जमानत",
                "ta": "பிணை",
                "te": "బెయిల్",
                "bn": "জামিন",
                "mr": "जामीन",
                "gu": "જામીન",
                "kn": "ಜಾಮೀನು",
                "ml": "ജാമ്യം",
                "pa": "ਜ਼ਮਾਨਤ",
                "en": "bail"
            },
            "criminal": {
                "hi": "आपराधिक",
                "ta": "குற்றவியல்",
                "te": "క్రిమినల్",
                "bn": "ফৌজদারি",
                "mr": "गुन्हेगारी",
                "gu": "ગુનાહિત",
                "kn": "ಅಪರಾಧ",
                "ml": "ക്രിമിനൽ",
                "pa": "ਅਪਰਾਧਿਕ",
                "en": "criminal"
            },
            "company": {
                "hi": "कंपनी",
                "ta": "நிறுவனம்",
                "te": "కంపెనీ",
                "bn": "কোম্পানি",
                "mr": "कंपनी",
                "gu": "કંપની",
                "kn": "ಕಂಪನಿ",
                "ml": "കമ്പനി",
                "pa": "ਕੰਪਨੀ",
                "en": "company"
            }
        }
    
    def _initialize_language_patterns(self) -> Dict[str, List[str]]:
        """Initialize language detection patterns"""
        return {
            "hi": ["है", "का", "की", "के", "में", "से", "को", "और", "या", "धारा", "कानून"],
            "ta": ["உள்ளது", "இல்", "அல்லது", "மற்றும்", "சட்டம்", "பிரிவு"],
            "te": ["ఉంది", "లో", "లేదా", "మరియు", "చట్టం", "సెక్షన్"],
            "bn": ["আছে", "এর", "বা", "এবং", "আইন", "ধারা"],
            "mr": ["आहे", "च्या", "किंवा", "आणि", "कायदा", "कलम"],
            "gu": ["છે", "ના", "અથવા", "અને", "કાયદો", "કલમ"],
            "kn": ["ಇದೆ", "ಅಥವಾ", "ಮತ್ತು", "ಕಾನೂನು", "ವಿಭಾಗ"],
            "ml": ["ഉണ്ട്", "അല്ലെങ്കിൽ", "കൂടാതെ", "നിയമം", "വകുപ്പ്"],
            "pa": ["ਹੈ", "ਦਾ", "ਜਾਂ", "ਅਤੇ", "ਕਾਨੂੰਨ", "ਧਾਰਾ"]
        }
    
    def detect_language(self, text: str) -> LanguageDetection:
        """Detect language of input text"""
        text_lower = text.lower()
        language_scores = {}
        
        # Check for English first (common case)
        english_words = ["what", "is", "the", "and", "or", "section", "law", "act", "under"]
        english_score = sum(1 for word in english_words if word in text_lower)
        language_scores["en"] = english_score
        
        # Check other languages
        for lang_code, patterns in self.language_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text)
            language_scores[lang_code] = score
        
        # Find best match
        if not language_scores or max(language_scores.values()) == 0:
            # Default to English if no patterns match
            detected_lang = "en"
            confidence = 0.5
        else:
            detected_lang = max(language_scores, key=language_scores.get)
            total_score = sum(language_scores.values())
            confidence = language_scores[detected_lang] / total_score if total_score > 0 else 0.5
        
        lang_info = self.supported_languages.get(detected_lang, self.supported_languages["en"])
        
        return LanguageDetection(
            language_code=detected_lang,
            language_name=lang_info["name"],
            confidence=confidence,
            script=lang_info["script"]
        )
    
    async def translate_query(self, query: str, source_lang: str, target_lang: str = "en") -> TranslationResult:
        """Translate legal query between languages"""
        
        if source_lang == target_lang:
            return TranslationResult(
                original_text=query,
                translated_text=query,
                source_language=source_lang,
                target_language=target_lang,
                confidence=1.0,
                translation_method="no_translation_needed"
            )
        
        # Try AI translation first
        if self.gemini_model:
            try:
                translated_text = await self._ai_translate(query, source_lang, target_lang)
                if translated_text:
                    return TranslationResult(
                        original_text=query,
                        translated_text=translated_text,
                        source_language=source_lang,
                        target_language=target_lang,
                        confidence=0.8,
                        translation_method="ai_translation"
                    )
            except Exception as e:
                logger.error(f"AI translation error: {e}")
        
        # Fallback to rule-based translation
        translated_text = self._rule_based_translate(query, source_lang, target_lang)
        
        return TranslationResult(
            original_text=query,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=0.6,
            translation_method="rule_based"
        )
    
    async def _ai_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """AI-powered translation using Gemini"""
        
        source_name = self.supported_languages.get(source_lang, {}).get("name", source_lang)
        target_name = self.supported_languages.get(target_lang, {}).get("name", target_lang)
        
        prompt = f"""Translate this legal query from {source_name} to {target_name}. 
Preserve legal terminology and maintain accuracy.

Text to translate: {text}

Provide only the translation, no explanations."""

        try:
            response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
            
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini translation error: {e}")
        
        return ""
    
    def _rule_based_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Rule-based translation using legal terminology dictionary"""
        
        if source_lang == "en" and target_lang in self.supported_languages:
            # English to Indian language
            translated = text
            for english_term, translations in self.legal_terms.items():
                if english_term in text.lower():
                    target_term = translations.get(target_lang, english_term)
                    translated = re.sub(r'\b' + re.escape(english_term) + r'\b', 
                                      target_term, translated, flags=re.IGNORECASE)
            return translated
        
        elif target_lang == "en":
            # Indian language to English
            translated = text
            for english_term, translations in self.legal_terms.items():
                source_term = translations.get(source_lang, "")
                if source_term and source_term in text:
                    translated = translated.replace(source_term, english_term)
            return translated
        
        # For other combinations, return original with note
        return f"[Translation needed from {source_lang} to {target_lang}] {text}"
    
    async def translate_response(self, response: str, target_lang: str) -> TranslationResult:
        """Translate legal response to target language"""
        
        if target_lang == "en":
            return TranslationResult(
                original_text=response,
                translated_text=response,
                source_language="en",
                target_language=target_lang,
                confidence=1.0,
                translation_method="no_translation_needed"
            )
        
        # AI translation for responses
        if self.gemini_model:
            try:
                target_name = self.supported_languages.get(target_lang, {}).get("name", target_lang)
                
                prompt = f"""Translate this legal response to {target_name}. 
Maintain legal accuracy and preserve section numbers, act names, and legal citations.
Keep the formatting and structure intact.

Response to translate:
{response}

Provide only the translation."""

                translated_response = await asyncio.to_thread(self.gemini_model.generate_content, prompt)
                
                if translated_response and hasattr(translated_response, 'text') and translated_response.text:
                    return TranslationResult(
                        original_text=response,
                        translated_text=translated_response.text,
                        source_language="en",
                        target_language=target_lang,
                        confidence=0.8,
                        translation_method="ai_translation"
                    )
            except Exception as e:
                logger.error(f"Response translation error: {e}")
        
        # Fallback to rule-based
        translated_text = self._rule_based_translate(response, "en", target_lang)
        
        return TranslationResult(
            original_text=response,
            translated_text=translated_text,
            source_language="en",
            target_language=target_lang,
            confidence=0.6,
            translation_method="rule_based"
        )
    
    def get_language_support_info(self) -> Dict[str, Any]:
        """Get information about supported languages"""
        return {
            "supported_languages": [
                {
                    "code": code,
                    "name": info["name"],
                    "native_name": info["native"],
                    "script": info["script"]
                }
                for code, info in self.supported_languages.items()
            ],
            "translation_methods": [
                "ai_translation",
                "rule_based",
                "legal_terminology_mapping"
            ],
            "features": [
                "Automatic language detection",
                "Legal terminology preservation",
                "Bidirectional translation",
                "Context-aware translation"
            ]
        }
    
    async def process_multilingual_query(self, query: str, preferred_response_lang: str = None) -> Dict[str, Any]:
        """Complete multilingual query processing pipeline"""
        
        # Step 1: Detect input language
        detection = self.detect_language(query)
        
        # Step 2: Translate query to English if needed
        if detection.language_code != "en":
            translation = await self.translate_query(query, detection.language_code, "en")
            english_query = translation.translated_text
        else:
            english_query = query
            translation = None
        
        # Step 3: Determine response language
        response_lang = preferred_response_lang or detection.language_code
        
        return {
            "original_query": query,
            "detected_language": {
                "code": detection.language_code,
                "name": detection.language_name,
                "confidence": detection.confidence,
                "script": detection.script
            },
            "english_query": english_query,
            "translation_used": translation is not None,
            "translation_details": translation.__dict__ if translation else None,
            "response_language": response_lang,
            "processing_notes": [
                f"Input detected as {detection.language_name}",
                f"Query translated to English" if translation else "No translation needed",
                f"Response will be in {self.supported_languages.get(response_lang, {}).get('name', response_lang)}"
            ]
        }

# Export the main classes
__all__ = ['MultilingualLegalEngine', 'LanguageDetection', 'TranslationResult']