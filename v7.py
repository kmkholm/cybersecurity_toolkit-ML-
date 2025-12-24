# -*- coding: utf-8 -*-
"""
Integrated Cybersecurity Toolkit v6.0 - ML Enhanced
Created on December 2025

Author: Dr. Mohammed Tawfik
Email: kmkhol01@gmail.com
License: Educational Use Only

NEW FEATURES v6.0:
✓ ML-Based Password Strength Analysis
✓ Colorful UI with Blue/Green/Purple themes
✓ Table displays for results
✓ Enhanced password analyzer with trained ML model

⚠️  WARNING: EDUCATIONAL USE ONLY - Unauthorized use is illegal!
"""

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, filedialog
import hashlib
import threading
import itertools
import re
import time
import json
import socket
import random
import string
import platform
import subprocess
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ML imports
try:
    import pandas as pd
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import shodan
try:
    import shodan
    SHODAN_AVAILABLE = True
except ImportError:
    SHODAN_AVAILABLE = False

# Additional imports for v6.0 features
try:
    import nmap
    NMAP_AVAILABLE = True
except ImportError:
    NMAP_AVAILABLE = False

try:
    import dns.resolver  
    DNS_AVAILABLE = True
except ImportError:
    DNS_AVAILABLE = False

try:
    import whois as whois_lib
    WHOIS_AVAILABLE = True
except ImportError:
    WHOIS_AVAILABLE = False

import ssl
import requests


# --- COLOR THEMES ---
COLORS = {
    'bg_dark': '#1a1a2e',
    'bg_medium': '#16213e',
    'bg_light': '#0f3460',
    'accent_blue': '#3498db',
    'accent_green': '#27ae60',
    'accent_purple': '#9b59b6',
    'accent_red': '#e74c3c',
    'accent_orange': '#e67e22',
    'accent_cyan': '#00d9ff',
    'text_white': '#ecf0f1',
    'text_gray': '#95a5a6',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'danger': '#e74c3c',
    'info': '#3498db'
}

# --- CONFIGURATION ---
MAX_WORDLIST_SIZE = 2000000
DEFAULT_CPU_THREADS = 8

class ApplicationSettings:
    """Global settings manager"""
    def __init__(self):
        self.cpu_workers = DEFAULT_CPU_THREADS
        self.gpu_enabled = False
        self.use_gpu = False
        self.max_brute_length = 8
        self.year_range_start = 2023
        self.year_range_end = 2026
        self.attack_paused = False
        self.attack_stopped = False
        self.use_common_combinations = True
        self.combo_min_length = 1
        self.combo_max_length = 8
        self.combo_charset_type = "letters+numbers"
        self.online_timeout = 5
        self.online_delay = 0.1
        self.online_threads = 4
        self.loaded_wordlist = None
        self.unlimited_mode = False
        self.max_attempts = 100000
        self.chatgpt_api_key = ""
        self.claude_api_key = ""
        self.ai_provider = "chatgpt"


SETTINGS = ApplicationSettings()

# Hash signatures
HASH_SIGNATURES = {
    'NetNTLMv2': (None, r'^[^:]+::[^:]+:[a-fA-F0-9]{16}:[a-fA-F0-9]{32}:[a-fA-F0-9]+$', 
                  'Windows network authentication. Requires specialized cracking.', 1),
    'NetNTLMv1': (None, r'^[^:]+::[^:]+:[a-fA-F0-9]{16}:[a-fA-F0-9]{48}:[a-fA-F0-9]{16}$',
                  'Legacy Windows network auth. Dictionary attack recommended.', 2),
    'SHA512': (128, r'^[a-fA-F0-9]{128}$', 'Medium speed, 128 hex. Dictionary highly preferred.', 3),
    'SHA256': (64, r'^[a-fA-F0-9]{64}$', 'Medium speed, 64 hex. Dictionary highly preferred.', 4),
    'SHA1': (40, r'^[a-fA-F0-9]{40}$', 'Fast, 40 hex. Dictionary/Brute-force viable.', 5),
    'MD5': (32, r'^[a-fA-F0-9]{32}$', 'Fast, 32 hex. Dictionary/Brute-force viable.', 6),
    'NTLM': (32, r'^[a-fA-F0-9]{32}$', 'Windows NTLM (MD4). Dictionary/Brute-force viable.', 7),
    'MySQL_OLD': (16, r'^[a-fA-F0-9]{16}$', 'Very weak. High-speed dictionary attack.', 8),
    'MD5(WordPress)': (34, r'^\$P\$[A-Za-z0-9\./]{31}$', 'Salted MD5. Targeted dictionary.', 9),
    'MD5(Joomla)': (49, r'^[a-fA-F0-9]{32}:[a-zA-Z0-9]{16}$', 'Salted MD5 + salt. Dictionary.', 10),
    'UNIX_BCRYPT': (60, r'^\$2[abyx]\$.{56}$', 'Slow. Targeted dictionary + GPU recommended.', 11),
    'UNIX_ARGON2': (None, r'^\$argon2[id]?\$v=\d+\$m=\d+,t=\d+,p=\d+\$.+$', 'Very slow. GPU required.', 12),
}

COMMON_PASSWORDS = [
    "password", "123456", "12345678", "12345", "qwerty", "abc123", "111111",
    "password123", "admin", "admin123", "root", "toor", "pass", "test",
    "welcome", "monkey", "dragon", "master", "letmein", "login", "princess",
    "1234", "1234567", "123456789", "password1", "qwerty123", "000000"
]

COMMON_APPENDS = ["123", "!", "@", "#", "$", "1", "12", "123!", "2024", "2025", "2026", "01", "!@#"]
COMMON_PREFIXES = ["!", "@", "#", "$", "admin", "user", "test"]

# =============================================================================
# ML PASSWORD STRENGTH ANALYZER
# =============================================================================

class MLPasswordAnalyzer:
    """ML-based password strength analyzer"""
    
    COMMON_WEAK = [
        "password", "123456", "12345678", "qwerty", "abc123", "admin",
        "iloveyou", "welcome", "letmein", "monkey", "dragon", "football"
    ]
    
    KEYBOARD_WALKS = ["qwerty", "asdfgh", "zxcvbn", "12345", "09876"]
    
    def __init__(self, model_path=None, features_path=None):
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        
        if model_path and features_path and ML_AVAILABLE:
            try:
                self.model = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.model_loaded = True
            except Exception as e:
                print(f"Model loading error: {e}")
    
    @staticmethod
    def char_space_size(pw):
        """Approximate character set size"""
        has_lower = any(c.islower() for c in pw)
        has_upper = any(c.isupper() for c in pw)
        has_digit = any(c.isdigit() for c in pw)
        has_symbol = any((not c.isalnum()) for c in pw)
        
        size = 0
        if has_lower: size += 26
        if has_upper: size += 26
        if has_digit: size += 10
        if has_symbol: size += 32
        return max(size, 1)
    
    @staticmethod
    def shannon_entropy_bits(pw):
        """Shannon entropy estimate"""
        cs = MLPasswordAnalyzer.char_space_size(pw)
        return len(pw) * math.log2(cs)
    
    @staticmethod
    def unique_ratio(pw):
        if not pw:
            return 0.0
        return len(set(pw)) / len(pw)
    
    @staticmethod
    def longest_run(pw):
        """Longest repeated-character run"""
        if not pw:
            return 0
        best, cur = 1, 1
        for i in range(1, len(pw)):
            if pw[i] == pw[i-1]:
                cur += 1
                best = max(best, cur)
            else:
                cur = 1
        return best
    
    @staticmethod
    def has_year_like(pw):
        """Detect year patterns"""
        for i in range(len(pw) - 3):
            chunk = pw[i:i+4]
            if chunk.isdigit():
                y = int(chunk)
                if 1900 <= y <= 2099:
                    return 1
        return 0
    
    @staticmethod
    def has_keyboard_walk(pw):
        low = pw.lower()
        return 1 if any(w in low for w in MLPasswordAnalyzer.KEYBOARD_WALKS) else 0
    
    @staticmethod
    def has_common_word(pw):
        low = pw.lower()
        return 1 if any(w in low for w in MLPasswordAnalyzer.COMMON_WEAK) else 0
    
    @staticmethod
    def sequential_digits_score(pw):
        """Detect sequential digits"""
        digits = [c for c in pw if c.isdigit()]
        s = "".join(digits)
        if len(s) < 4:
            return 0
        for i in range(len(s) - 3):
            chunk = s[i:i+4]
            if chunk in "0123456789" or chunk in "9876543210":
                return 1
        return 0
    
    @staticmethod
    def extract_features(pw):
        """Extract all features from password"""
        return {
            'length': len(pw),
            'has_lower': int(any(c.islower() for c in pw)),
            'has_upper': int(any(c.isupper() for c in pw)),
            'has_digit': int(any(c.isdigit() for c in pw)),
            'has_symbol': int(any((not c.isalnum()) for c in pw)),
            'unique_ratio': round(MLPasswordAnalyzer.unique_ratio(pw), 4),
            'longest_run': MLPasswordAnalyzer.longest_run(pw),
            'has_year_like': MLPasswordAnalyzer.has_year_like(pw),
            'has_common_word': MLPasswordAnalyzer.has_common_word(pw),
            'has_keyboard_walk': MLPasswordAnalyzer.has_keyboard_walk(pw),
            'has_sequential_digits': MLPasswordAnalyzer.sequential_digits_score(pw),
            'entropy_bits': round(MLPasswordAnalyzer.shannon_entropy_bits(pw), 3)
        }
    
    @staticmethod
    def strength_score(pw):
        """Calculate strength score"""
        ent = MLPasswordAnalyzer.shannon_entropy_bits(pw)
        
        penalty = 0.0
        penalty += 18.0 * MLPasswordAnalyzer.has_common_word(pw)
        penalty += 10.0 * MLPasswordAnalyzer.has_year_like(pw)
        penalty += 8.0  * MLPasswordAnalyzer.has_keyboard_walk(pw)
        penalty += 8.0  * MLPasswordAnalyzer.sequential_digits_score(pw)
        
        ur = MLPasswordAnalyzer.unique_ratio(pw)
        if ur < 0.6:
            penalty += 10.0
        
        lr = MLPasswordAnalyzer.longest_run(pw)
        if lr >= 3:
            penalty += 6.0
        
        return max(ent - penalty, 0.0)
    
    @staticmethod
    def rule_based_label(score):
        """Rule-based classification"""
        if score < 35:
            return "Weak"
        elif score < 70:
            return "Medium"
        else:
            return "Strong"
    
    def analyze(self, password):
        """Analyze password strength"""
        features = self.extract_features(password)
        score = self.strength_score(password)
        features['strength_score'] = round(score, 3)
        
        # Try ML prediction if model is loaded
        if self.model_loaded and self.feature_names:
            try:
                df = pd.DataFrame([features])
                df = df[self.feature_names]
                prediction = self.model.predict(df)[0]
                proba = self.model.predict_proba(df)[0]
                features['ml_prediction'] = prediction
                features['ml_confidence'] = {
                    'Weak': round(proba[2], 3),
                    'Medium': round(proba[1], 3),
                    'Strong': round(proba[0], 3)
                }
            except Exception as e:
                features['ml_prediction'] = "N/A"
                features['ml_confidence'] = {}
        
        # Rule-based classification as fallback
        features['rule_based_label'] = self.rule_based_label(score)
        
        return features


# =============================================================================
# ML HASH IDENTIFIER
# =============================================================================

class MLHashIdentifier:
    """ML-based hash type identifier"""
    
    def __init__(self, model_path=None, features_path=None):
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        
        if model_path and features_path and ML_AVAILABLE:
            try:
                self.model = joblib.load(model_path)
                with open(features_path, 'r') as f:
                    self.feature_names = json.load(f)
                self.model_loaded = True
            except Exception as e:
                print(f"Hash model loading error: {e}")
    
    @staticmethod
    def extract_hash_features(hash_value):
        """Extract features from hash for ML"""
        features = {}
        
        # Basic features
        features['length'] = len(hash_value)
        features['has_dollar'] = int('$' in hash_value)
        features['has_colon'] = int(':' in hash_value)
        features['has_uppercase'] = int(any(c.isupper() for c in hash_value))
        features['has_lowercase'] = int(any(c.islower() for c in hash_value))
        features['has_digit'] = int(any(c.isdigit() for c in hash_value))
        features['has_special'] = int(any(c in './$@+=' for c in hash_value))
        
        # Character distribution
        features['hex_only'] = int(all(c in '0123456789abcdefABCDEF' for c in hash_value.replace('$', '').replace(':', '')))
        features['alphanumeric_only'] = int(all(c.isalnum() or c in '$:' for c in hash_value))
        features['has_slash'] = int('/' in hash_value)
        features['has_plus'] = int('+' in hash_value)
        features['has_equals'] = int('=' in hash_value)
        
        # Pattern features
        features['starts_with_dollar'] = int(hash_value.startswith('$'))
        features['dollar_count'] = hash_value.count('$')
        features['colon_count'] = hash_value.count(':')
        
        # Statistical features
        if len(hash_value) > 0:
            features['unique_char_ratio'] = len(set(hash_value)) / len(hash_value)
            features['digit_ratio'] = sum(c.isdigit() for c in hash_value) / len(hash_value)
            features['alpha_ratio'] = sum(c.isalpha() for c in hash_value) / len(hash_value)
        else:
            features['unique_char_ratio'] = 0
            features['digit_ratio'] = 0
            features['alpha_ratio'] = 0
        
        # Length categories
        features['is_short'] = int(len(hash_value) <= 20)
        features['is_medium'] = int(20 < len(hash_value) <= 50)
        features['is_long'] = int(50 < len(hash_value) <= 80)
        features['is_very_long'] = int(len(hash_value) > 80)
        
        # Specific length markers
        features['is_len_16'] = int(len(hash_value) == 16)
        features['is_len_32'] = int(len(hash_value) == 32)
        features['is_len_40'] = int(len(hash_value) == 40)
        features['is_len_56'] = int(len(hash_value) == 56)
        features['is_len_64'] = int(len(hash_value) == 64)
        features['is_len_96'] = int(len(hash_value) == 96)
        features['is_len_128'] = int(len(hash_value) == 128)
        
        # Format patterns
        features['has_prefix_2a'] = int(hash_value.startswith('$2a$'))
        features['has_prefix_2b'] = int(hash_value.startswith('$2b$'))
        features['has_prefix_P'] = int(hash_value.startswith('$P$'))
        features['has_prefix_H'] = int(hash_value.startswith('$H$'))
        
        return features
    
    def identify(self, hash_value):
        """Identify hash type using ML model"""
        features = self.extract_hash_features(hash_value)
        
        if self.model_loaded and self.feature_names:
            try:
                # Prepare features
                df = pd.DataFrame([features])
                df = df[self.feature_names]
                
                # Predict
                prediction = self.model.predict(df)[0]
                probabilities = self.model.predict_proba(df)[0]
                
                # Get top predictions
                classes = self.model.named_steps['model'].classes_
                sorted_indices = probabilities.argsort()[::-1]
                
                top_predictions = []
                for idx in sorted_indices[:3]:
                    top_predictions.append({
                        'type': classes[idx],
                        'probability': float(probabilities[idx])
                    })
                
                # Determine confidence
                max_prob = probabilities.max()
                if max_prob >= 0.85:
                    confidence = 'High'
                elif max_prob >= 0.60:
                    confidence = 'Medium'
                else:
                    confidence = 'Low'
                
                return {
                    'type': prediction,
                    'confidence': confidence,
                    'probability': float(max_prob),
                    'alternatives': [p for p in top_predictions[1:3]],
                    'ml_prediction': True
                }
            except Exception as e:
                print(f"ML prediction error: {e}")
        
        # Fallback to rule-based
        return self.rule_based_identify(hash_value)
    
    def rule_based_identify(self, hash_value):
        """Fallback rule-based identification"""
        hash_type, recommendation = identify_hash(hash_value)
        
        # Estimate confidence
        if hash_type in ['MD5', 'SHA1', 'SHA256', 'SHA512', 'Bcrypt', 'WordPress']:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return {
            'type': hash_type,
            'confidence': confidence,
            'probability': 0.5,
            'alternatives': [],
            'ml_prediction': False,
            'recommendation': recommendation
        }


# =============================================================================
# HASH IDENTIFICATION & UTILITIES
# =============================================================================

def identify_hash(hash_string, context_hints=None):
    """Hash identification"""
    hash_length = len(hash_string)
    possible_matches = []
    
    for name, (length, pattern, recommendation, priority) in HASH_SIGNATURES.items():
        if re.match(pattern, hash_string):
            if length is None or hash_length == length:
                possible_matches.append((name, recommendation, priority))
    
    if context_hints and len(possible_matches) > 1:
        source = context_hints.get('source', '').lower()
        if source == 'windows':
            for match in possible_matches:
                if 'NTLM' in match[0]:
                    return match[0], match[1]
    
    if possible_matches:
        possible_matches.sort(key=lambda x: x[2])
        return possible_matches[0][0], possible_matches[0][1]
    
    if hash_string.count('$') >= 2 or hash_string.count(':') >= 1:
        return 'Custom-Salted', "Custom salted format. Full context required."
    
    if hash_length > 10 and re.match(r'^[a-zA-Z0-9+/=]+$', hash_string):
        return 'Unknown-Complex-Base64', "Base64-encoded. Dictionary recommended."
    
    return 'Unknown-Brute', "Hash structure not recognized. Try dictionary/brute-force."

# --- WORDLIST GENERATION ---

def apply_leetspeak(word):
    """Leetspeak mutations"""
    leetspeak_map = {
        'a': ['4', '@'], 'e': ['3'], 'i': ['1', '!'], 
        'o': ['0'], 's': ['$', '5'], 't': ['7'], 
        'l': ['1'], 'g': ['9'], 'b': ['8']
    }
    mutations = {word}
    
    for char, replacements in leetspeak_map.items():
        if char in word.lower():
            for replacement in replacements:
                mutations.add(word.replace(char, replacement))
                mutations.add(word.replace(char.upper(), replacement))
    
    return mutations

def generate_charset_from_type(charset_type):
    """Generate charset from type"""
    if charset_type == "letters":
        return string.ascii_lowercase
    elif charset_type == "LETTERS":
        return string.ascii_uppercase
    elif charset_type == "numbers":
        return string.digits
    elif charset_type == "letters+numbers":
        return string.ascii_lowercase + string.digits
    elif charset_type == "LETTERS+numbers":
        return string.ascii_uppercase + string.digits
    elif charset_type == "Letters+Numbers":
        return string.ascii_letters + string.digits
    elif charset_type == "all":
        return string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>?"
    else:
        return string.printable.strip()

def contextual_wordlist_generator(
    base_words,
    years_range=(2020, 2026),
    common_appends=COMMON_APPENDS,
    common_prefixes=COMMON_PREFIXES,
    apply_leet=True,
    include_basic_combinations=True,
    max_length=None
):
    """Enhanced contextual wordlist generator"""
    wordlist = set()
    
    for word in base_words:
        wordlist.add(word)
        wordlist.add(word.lower())
        wordlist.add(word.upper())
        wordlist.add(word.capitalize())
        
        for year in range(years_range[0], years_range[1] + 1):
            wordlist.add(f"{word}{year}")
            wordlist.add(f"{word}{str(year)[2:]}")
            wordlist.add(f"{year}{word}")
        
        for append in common_appends:
            wordlist.add(f"{word}{append}")
        
        for prefix in common_prefixes:
            wordlist.add(f"{prefix}{word}")
        
        if apply_leet:
            for variant in apply_leetspeak(word):
                wordlist.add(variant)
                for append in common_appends[:5]:
                    wordlist.add(f"{variant}{append}")
        
        if include_basic_combinations:
            for other in base_words[:10]:
                if word != other:
                    wordlist.add(f"{word}{other}")
                    wordlist.add(f"{word}{other.capitalize()}")
    
    if max_length:
        wordlist = {w for w in wordlist if len(w) <= max_length}
    
    return sorted(wordlist)

def smart_combination_generator(min_len, max_len, charset_type="letters+numbers", max_attempts=100000):
    """Smart combination generator"""
    charset = generate_charset_from_type(charset_type)
    count = 0
    
    for length in range(min_len, max_len + 1):
        for combination in itertools.product(charset, repeat=length):
            if count >= max_attempts:
                return
            yield ''.join(combination)
            count += 1


# =============================================================================
# PASSWORD CRACKING ENGINE
# =============================================================================

class PasswordCracker:
    """Multi-threaded password cracking engine"""
    
    HASH_FUNCTIONS = {
        'MD5': lambda x: hashlib.md5(x.encode()).hexdigest(),
        'SHA1': lambda x: hashlib.sha1(x.encode()).hexdigest(),
        'SHA256': lambda x: hashlib.sha256(x.encode()).hexdigest(),
        'SHA512': lambda x: hashlib.sha512(x.encode()).hexdigest(),
        'NTLM': lambda x: hashlib.new('md4', x.encode('utf-16le')).hexdigest(),
    }
    
    @staticmethod
    def crack_hash(target_hash, hash_type, wordlist=None, use_combinations=False, 
                   combo_min=1, combo_max=8, charset="letters+numbers", 
                   max_attempts=100000, progress_callback=None):
        """Main hash cracking function"""
        
        if hash_type not in PasswordCracker.HASH_FUNCTIONS:
            return {'success': False, 'error': f'Unsupported hash type: {hash_type}'}
        
        hash_func = PasswordCracker.HASH_FUNCTIONS[hash_type]
        target_hash = target_hash.lower()
        attempts = 0
        start_time = time.time()
        
        # Try common passwords first
        for password in COMMON_PASSWORDS:
            if SETTINGS.attack_stopped:
                return {'success': False, 'stopped': True, 'attempts': attempts}
            
            attempts += 1
            if hash_func(password).lower() == target_hash:
                elapsed = time.time() - start_time
                return {
                    'success': True,
                    'password': password,
                    'attempts': attempts,
                    'time': round(elapsed, 2),
                    'rate': int(attempts / elapsed) if elapsed > 0 else 0
                }
            
            if progress_callback and attempts % 100 == 0:
                progress_callback(attempts, password)
        
        # Wordlist attack
        if wordlist:
            for password in wordlist:
                if SETTINGS.attack_stopped:
                    return {'success': False, 'stopped': True, 'attempts': attempts}
                
                while SETTINGS.attack_paused:
                    time.sleep(0.1)
                
                attempts += 1
                if hash_func(password).lower() == target_hash:
                    elapsed = time.time() - start_time
                    return {
                        'success': True,
                        'password': password,
                        'attempts': attempts,
                        'time': round(elapsed, 2),
                        'rate': int(attempts / elapsed) if elapsed > 0 else 0
                    }
                
                if progress_callback and attempts % 1000 == 0:
                    progress_callback(attempts, password)
        
        # Combination/brute-force attack
        if use_combinations:
            for password in smart_combination_generator(combo_min, combo_max, charset, max_attempts):
                if SETTINGS.attack_stopped:
                    return {'success': False, 'stopped': True, 'attempts': attempts}
                
                while SETTINGS.attack_paused:
                    time.sleep(0.1)
                
                attempts += 1
                if hash_func(password).lower() == target_hash:
                    elapsed = time.time() - start_time
                    return {
                        'success': True,
                        'password': password,
                        'attempts': attempts,
                        'time': round(elapsed, 2),
                        'rate': int(attempts / elapsed) if elapsed > 0 else 0
                    }
                
                if progress_callback and attempts % 1000 == 0:
                    progress_callback(attempts, password)
        
        elapsed = time.time() - start_time
        return {
            'success': False,
            'attempts': attempts,
            'time': round(elapsed, 2),
            'rate': int(attempts / elapsed) if elapsed > 0 else 0
        }


# =============================================================================
# NETWORK TOOLS
# =============================================================================

class NetworkTools:
    """Network diagnostic tools"""
    
    @staticmethod
    def ping(host, count=4, timeout=2):
        """Ping a host"""
        try:
            param = '-n' if platform.system().lower() == 'windows' else '-c'
            timeout_param = '-w' if platform.system().lower() == 'windows' else '-W'
            
            cmd = ['ping', param, str(count), timeout_param, str(timeout * 1000 if platform.system().lower() == 'windows' else timeout), host]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * count + 5)
            
            return {
                'success': True,
                'output': result.stdout,
                'returncode': result.returncode,
                'reachable': result.returncode == 0
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def traceroute(host, max_hops=30):
        """Traceroute to host"""
        try:
            system = platform.system().lower()
            
            if system == 'windows':
                cmd = ['tracert', '-h', str(max_hops), host]
            elif system == 'darwin':  # macOS
                cmd = ['traceroute', '-m', str(max_hops), host]
            else:  # Linux
                # Try traceroute first, fall back to tracepath if not available
                try:
                    subprocess.run(['traceroute', '--version'], capture_output=True, timeout=1)
                    cmd = ['traceroute', '-m', str(max_hops), host]
                except:
                    # Use tracepath as fallback (usually available without sudo)
                    cmd = ['tracepath', host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Check if command failed due to permissions or not found
            if result.returncode != 0 and not result.stdout and 'not found' in result.stderr.lower():
                return {
                    'success': False,
                    'error': f"Traceroute not found. Install with:\n  Ubuntu/Debian: sudo apt-get install traceroute\n  CentOS/RHEL: sudo yum install traceroute\n  macOS: Usually pre-installed\n  Windows: Use 'tracert' command"
                }
            
            return {
                'success': True,
                'output': result.stdout if result.stdout else result.stderr,
                'returncode': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Traceroute timed out (120s limit)'}
        except FileNotFoundError:
            system = platform.system().lower()
            if system == 'linux':
                return {'success': False, 'error': 'Traceroute not found. Install: sudo apt-get install traceroute'}
            else:
                return {'success': False, 'error': 'Traceroute command not found'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def port_scan(host, ports):
        """Simple port scanner"""
        results = {}
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                results[port] = 'Open' if result == 0 else 'Closed'
                sock.close()
            except Exception as e:
                results[port] = f'Error: {str(e)}'
        
        return {'success': True, 'results': results}


class ShodanSearcher:
    """Shodan API integration"""
    
    @staticmethod
    def search(api_key, query, limit=10):
        """Search Shodan"""
        if not SHODAN_AVAILABLE:
            return {'success': False, 'error': 'Shodan library not installed'}
        
        try:
            api = shodan.Shodan(api_key)
            results = api.search(query, limit=limit)
            
            devices = []
            for result in results['matches']:
                devices.append({
                    'ip': result.get('ip_str', 'N/A'),
                    'port': result.get('port', 'N/A'),
                    'org': result.get('org', 'N/A'),
                    'location': f"{result.get('location', {}).get('city', 'N/A')}, {result.get('location', {}).get('country_name', 'N/A')}",
                    'banner': result.get('data', 'N/A')[:200]
                })
            
            return {
                'success': True,
                'total': results['total'],
                'devices': devices
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class NmapScanner:
    """Nmap integration"""
    
    @staticmethod
    def scan(target, arguments='-sV', ports=None):
        """Perform nmap scan with custom arguments"""
        if not NMAP_AVAILABLE:
            return {'success': False, 'error': 'python-nmap not installed'}
        
        try:
            nm = nmap.PortScanner()
            
            # Scan with custom arguments
            if ports:
                nm.scan(target, ports=ports, arguments=arguments)
            else:
                nm.scan(target, arguments=arguments)
            
            results = {}
            for host in nm.all_hosts():
                results[host] = {
                    'state': nm[host].state(),
                    'protocols': {},
                    'hostname': nm[host].hostname(),
                }
                
                # OS detection if available
                if 'osmatch' in nm[host]:
                    results[host]['osmatch'] = nm[host]['osmatch']
                
                for proto in nm[host].all_protocols():
                    results[host]['protocols'][proto] = {}
                    ports = nm[host][proto].keys()
                    for port in ports:
                        port_info = nm[host][proto][port]
                        results[host]['protocols'][proto][port] = {
                            'state': port_info['state'],
                            'name': port_info.get('name', 'unknown'),
                            'product': port_info.get('product', ''),
                            'version': port_info.get('version', ''),
                            'extrainfo': port_info.get('extrainfo', ''),
                        }
            
            return {'success': True, 'results': results}
        except nmap.PortScannerError as e:
            return {'success': False, 'error': f'Nmap error: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class WhoisLookup:
    """WHOIS lookup"""
    
    @staticmethod
    def lookup(domain):
        """Perform WHOIS lookup"""
        if not WHOIS_AVAILABLE:
            return {'success': False, 'error': 'python-whois not installed'}
        
        try:
            w = whois_lib.whois(domain)
            return {
                'success': True,
                'domain': w.domain_name,
                'registrar': w.registrar,
                'creation_date': str(w.creation_date),
                'expiration_date': str(w.expiration_date),
                'name_servers': w.name_servers,
                'status': w.status,
                'emails': w.emails,
                'raw': w.text
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class DNSAnalyzer:
    """DNS analysis"""
    
    @staticmethod
    def analyze(domain):
        """Perform DNS analysis"""
        if not DNS_AVAILABLE:
            return {'success': False, 'error': 'dnspython not installed'}
        
        try:
            results = {}
            record_types = ['A', 'AAAA', 'MX', 'NS', 'TXT', 'CNAME', 'SOA']
            
            for rtype in record_types:
                try:
                    answers = dns.resolver.resolve(domain, rtype)
                    results[rtype] = [str(rdata) for rdata in answers]
                except:
                    results[rtype] = []
            
            return {'success': True, 'results': results}
        except Exception as e:
            return {'success': False, 'error': str(e)}


class SSLAnalyzer:
    """SSL/TLS certificate analyzer"""
    
    @staticmethod
    def analyze(hostname, port=443):
        """Analyze SSL certificate"""
        try:
            context = ssl.create_default_context()
            with socket.create_connection((hostname, port), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    cipher = ssock.cipher()
                    version = ssock.version()
            
            return {
                'success': True,
                'subject': dict(x[0] for x in cert['subject']),
                'issuer': dict(x[0] for x in cert['issuer']),
                'version': cert['version'],
                'serial': cert['serialNumber'],
                'not_before': cert['notBefore'],
                'not_after': cert['notAfter'],
                'cipher': cipher[0] if cipher else 'N/A',
                'tls_version': version
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


class AIAnalyzer:
    """AI-powered vulnerability analysis"""
    
    @staticmethod
    def analyze_with_chatgpt(api_key, scan_results):
        """Analyze with ChatGPT"""
        try:
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""Analyze these cybersecurity scan results and provide:
1. Security vulnerabilities identified
2. Risk assessment (Critical/High/Medium/Low)
3. Specific recommendations for hardening
4. Compliance considerations

Scan Results:
{scan_results}"""
            
            data = {
                'model': 'gpt-4',
                'messages': [
                    {'role': 'system', 'content': 'You are a cybersecurity expert analyzing scan results.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 2000
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'analysis': result['choices'][0]['message']['content']
                }
            else:
                return {'success': False, 'error': f'API Error: {response.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def analyze_with_claude(api_key, scan_results):
        """Analyze with Claude"""
        try:
            headers = {
                'x-api-key': api_key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json'
            }
            
            prompt = f"""Analyze these cybersecurity scan results and provide:
1. Security vulnerabilities identified
2. Risk assessment (Critical/High/Medium/Low)
3. Specific recommendations for hardening
4. Compliance considerations

Scan Results:
{scan_results}"""
            
            data = {
                'model': 'claude-3-opus-20240229',
                'max_tokens': 2000,
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }
            
            response = requests.post(
                'https://api.anthropic.com/v1/messages',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    'success': True,
                    'analysis': result['content'][0]['text']
                }
            else:
                return {'success': False, 'error': f'API Error: {response.status_code}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}


# =============================================================================
# GUI APPLICATION
# =============================================================================

class CyberSecurityToolkit_GUI:
    """Main GUI application"""
    
    def __init__(self, master):
        self.master = master
        self.master.title("🛡️ Cybersecurity Toolkit v6.0 - ML Enhanced")
        self.master.geometry("1400x900")
        self.master.configure(bg=COLORS['bg_dark'])
        
        # Configure style
        self.setup_styles()
        
        # ML Analyzers
        self.ml_analyzer = MLPasswordAnalyzer()
        self.ml_hash_identifier = MLHashIdentifier()
        
        # Create notebook
        self.notebook = ttk.Notebook(self.master, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_ml_password_tab()
        self.create_hash_cracker_tab()
        self.create_network_tools_tab()
        self.create_online_attacks_tab()
        self.create_vulnerability_scanner_tab()
        self.create_shodan_tab()
        self.create_nmap_tab()
        self.create_whois_tab()
        self.create_dns_tab()
        self.create_ssl_tab()
        self.create_ai_tab()
        
        # Status bar
        self.create_status_bar()
    
    def setup_styles(self):
        """Setup custom styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Notebook style
        style.configure('Custom.TNotebook', background=COLORS['bg_dark'], borderwidth=0)
        style.configure('Custom.TNotebook.Tab', 
                       background=COLORS['bg_medium'],
                       foreground=COLORS['text_white'],
                       padding=[15, 8],
                       font=('Arial', 10, 'bold'))
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', COLORS['accent_blue'])],
                 foreground=[('selected', COLORS['text_white'])])
        
        # Frame style
        style.configure('Custom.TFrame', background=COLORS['bg_medium'])
        style.configure('TLabelframe', background=COLORS['bg_medium'], 
                       foreground=COLORS['text_white'], borderwidth=2)
        style.configure('TLabelframe.Label', background=COLORS['bg_medium'],
                       foreground=COLORS['accent_cyan'], font=('Arial', 11, 'bold'))
        
        # Treeview style
        style.configure('Custom.Treeview',
                       background=COLORS['bg_dark'],
                       foreground=COLORS['text_white'],
                       fieldbackground=COLORS['bg_dark'],
                       borderwidth=0)
        style.configure('Custom.Treeview.Heading',
                       background=COLORS['accent_blue'],
                       foreground=COLORS['text_white'],
                       font=('Arial', 10, 'bold'))
        style.map('Custom.Treeview',
                 background=[('selected', COLORS['accent_purple'])])
    
    def create_status_bar(self):
        """Create status bar"""
        status_frame = tk.Frame(self.master, bg=COLORS['bg_light'], height=30)
        status_frame.pack(side='bottom', fill='x')
        
        self.status_label = tk.Label(status_frame, text="✓ Ready", 
                                     bg=COLORS['bg_light'], fg=COLORS['success'],
                                     font=('Arial', 9), anchor='w')
        self.status_label.pack(side='left', padx=10)
        
        # ML Status
        ml_status = "✓ ML Available" if ML_AVAILABLE else "⚠ ML Not Available"
        ml_color = COLORS['success'] if ML_AVAILABLE else COLORS['warning']
        tk.Label(status_frame, text=ml_status, bg=COLORS['bg_light'], 
                fg=ml_color, font=('Arial', 9)).pack(side='right', padx=10)
    
    # =========================================================================
    # ML PASSWORD STRENGTH ANALYZER TAB
    # =========================================================================
    
    def create_ml_password_tab(self):
        """Create ML password strength analyzer tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔐 ML Password & Hash Analyzer")
        
        # Header
        header = tk.Frame(tab, bg=COLORS['accent_purple'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🤖 Machine Learning Password & Hash Analyzer",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_purple'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Create main container with two columns
        main_container = tk.Frame(tab, bg=COLORS['bg_dark'])
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Left column - Password Analysis (60%)
        left_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        left_frame.pack(side='left', fill='both', expand=True, padx=5)
        
        # Right column - Hash Identification (40%)
        right_frame = tk.Frame(main_container, bg=COLORS['bg_dark'])
        right_frame.pack(side='right', fill='both', expand=False, padx=5)
        
        # Setup left side (Password Analysis)
        self.setup_password_analysis(left_frame)
        
        # Setup right side (Hash Identification)
        self.setup_hash_identification(right_frame)
    
    def setup_password_analysis(self, parent):
        """Setup password analysis section"""
    def setup_password_analysis(self, parent):
        """Setup password analysis section"""
        
        # Title
        title_frame = tk.Frame(parent, bg=COLORS['accent_green'], height=40)
        title_frame.pack(fill='x', pady=(0, 5))
        tk.Label(title_frame, text="🔐 PASSWORD STRENGTH ANALYSIS",
                font=('Arial', 12, 'bold'), bg=COLORS['accent_green'],
                fg=COLORS['text_white']).pack(pady=8)
        
        # Model loading section
        model_frame = ttk.LabelFrame(parent, text="⚙️ ML Model Configuration", padding=10)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(model_frame, text="Model File:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.ml_model_path = tk.Entry(model_frame, width=35, font=('Arial', 8))
        self.ml_model_path.grid(row=0, column=1, padx=5, pady=3)
        self.ml_model_path.insert(0, "pw_strength_model.pkl")
        
        tk.Button(model_frame, text="📁", command=self.browse_model_file,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 8, 'bold'),
                 width=3).grid(row=0, column=2, padx=2)
        
        tk.Label(model_frame, text="Features File:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.ml_features_path = tk.Entry(model_frame, width=35, font=('Arial', 8))
        self.ml_features_path.grid(row=1, column=1, padx=5, pady=3)
        self.ml_features_path.insert(0, "pw_strength_features.json")
        
        tk.Button(model_frame, text="📁", command=self.browse_features_file,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 8, 'bold'),
                 width=3).grid(row=1, column=2, padx=2)
        
        btn_frame_model = tk.Frame(model_frame, bg=COLORS['bg_medium'])
        btn_frame_model.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        
        tk.Button(btn_frame_model, text="🔄 Load Model", command=self.load_ml_model,
                 bg=COLORS['success'], fg='white', font=('Arial', 9, 'bold'),
                 height=1).pack(fill='x', padx=2)
        
        self.ml_model_status = tk.Label(model_frame, text="⚠ Model not loaded",
                                       bg=COLORS['bg_medium'], fg=COLORS['warning'],
                                       font=('Arial', 8, 'bold'))
        self.ml_model_status.grid(row=3, column=0, columnspan=3, pady=2)
        
        # Input section
        input_frame = ttk.LabelFrame(parent, text="🔍 Password Analysis", padding=10)
        input_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        tk.Label(input_frame, text="Enter Password(s) - one per line:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(anchor='w')
        
        self.ml_password_input = scrolledtext.ScrolledText(input_frame, height=5,
                                                           font=('Arial', 10),
                                                           bg='white', fg='black',
                                                           wrap=tk.WORD)
        self.ml_password_input.pack(fill='both', expand=True, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_medium'])
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="🔍 Analyze Passwords", command=self.analyze_passwords_ml,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 10, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=2)
        
        tk.Button(btn_frame, text="🧪 Generate Test", command=self.generate_test_dataset,
                 bg=COLORS['accent_orange'], fg='white', font=('Arial', 10, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=2)
        
        tk.Button(btn_frame, text="🧹 Clear All", command=self.clear_ml_results,
                 bg=COLORS['accent_red'], fg='white', font=('Arial', 10, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=2)
        
        # Results table
        results_frame = ttk.LabelFrame(parent, text="📊 Analysis Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create Treeview with scrollbars
        tree_scroll_y = ttk.Scrollbar(results_frame, orient='vertical')
        tree_scroll_y.pack(side='right', fill='y')
        
        tree_scroll_x = ttk.Scrollbar(results_frame, orient='horizontal')
        tree_scroll_x.pack(side='bottom', fill='x')
        
        columns = ('Password', 'Length', 'Entropy', 'Score', 'Rule-Based', 'ML Prediction', 'Confidence')
        self.ml_results_tree = ttk.Treeview(results_frame, columns=columns, show='headings',
                                           style='Custom.Treeview',
                                           yscrollcommand=tree_scroll_y.set,
                                           xscrollcommand=tree_scroll_x.set,
                                           height=8)
        
        tree_scroll_y.config(command=self.ml_results_tree.yview)
        tree_scroll_x.config(command=self.ml_results_tree.xview)
        
        # Configure columns
        self.ml_results_tree.heading('Password', text='Password')
        self.ml_results_tree.heading('Length', text='Len')
        self.ml_results_tree.heading('Entropy', text='Entropy')
        self.ml_results_tree.heading('Score', text='Score')
        self.ml_results_tree.heading('Rule-Based', text='Rule')
        self.ml_results_tree.heading('ML Prediction', text='ML Pred')
        self.ml_results_tree.heading('Confidence', text='Conf%')
        
        self.ml_results_tree.column('Password', width=150)
        self.ml_results_tree.column('Length', width=50, anchor='center')
        self.ml_results_tree.column('Entropy', width=70, anchor='center')
        self.ml_results_tree.column('Score', width=60, anchor='center')
        self.ml_results_tree.column('Rule-Based', width=70, anchor='center')
        self.ml_results_tree.column('ML Prediction', width=80, anchor='center')
        self.ml_results_tree.column('Confidence', width=70, anchor='center')
        
        self.ml_results_tree.pack(fill='both', expand=True)
        
        # Add row colors
        self.ml_results_tree.tag_configure('weak', background='#e74c3c', foreground='white')
        self.ml_results_tree.tag_configure('medium', background='#f39c12', foreground='white')
        self.ml_results_tree.tag_configure('strong', background='#27ae60', foreground='white')
        
        # Detailed info section
        detail_frame = ttk.LabelFrame(parent, text="📋 Detailed Information", padding=5)
        detail_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.ml_detail_text = scrolledtext.ScrolledText(detail_frame, height=6,
                                                        bg=COLORS['bg_dark'],
                                                        fg=COLORS['accent_cyan'],
                                                        font=('Courier', 8))
        self.ml_detail_text.pack(fill='both', expand=True)
        
        # Bind selection event
        self.ml_results_tree.bind('<<TreeviewSelect>>', self.on_password_select)
    
    def setup_hash_identification(self, parent):
        """Setup hash identification section"""
        
        # Make right panel fixed width
        parent.config(width=450)
        
        # Title
        title_frame = tk.Frame(parent, bg=COLORS['accent_blue'], height=40)
        title_frame.pack(fill='x', pady=(0, 5))
        tk.Label(title_frame, text="🔍 ML HASH IDENTIFIER",
                font=('Arial', 12, 'bold'), bg=COLORS['accent_blue'],
                fg=COLORS['text_white']).pack(pady=8)
        
        # ML Model loading section
        hash_model_frame = ttk.LabelFrame(parent, text="⚙️ Hash ML Model", padding=10)
        hash_model_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(hash_model_frame, text="Model:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.hash_model_path = tk.Entry(hash_model_frame, width=25, font=('Arial', 8))
        self.hash_model_path.grid(row=0, column=1, padx=5, pady=3)
        self.hash_model_path.insert(0, "hash_identifier_model.pkl")
        
        tk.Button(hash_model_frame, text="📁", command=self.browse_hash_model_file,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 8, 'bold'),
                 width=3).grid(row=0, column=2, padx=2)
        
        tk.Label(hash_model_frame, text="Features:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.hash_features_path = tk.Entry(hash_model_frame, width=25, font=('Arial', 8))
        self.hash_features_path.grid(row=1, column=1, padx=5, pady=3)
        self.hash_features_path.insert(0, "hash_identifier_features.json")
        
        tk.Button(hash_model_frame, text="📁", command=self.browse_hash_features_file,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 8, 'bold'),
                 width=3).grid(row=1, column=2, padx=2)
        
        btn_frame_hash_model = tk.Frame(hash_model_frame, bg=COLORS['bg_medium'])
        btn_frame_hash_model.grid(row=2, column=0, columnspan=3, pady=5, sticky='ew')
        
        tk.Button(btn_frame_hash_model, text="🔄 Load Hash Model", command=self.load_hash_ml_model,
                 bg=COLORS['success'], fg='white', font=('Arial', 9, 'bold'),
                 height=1).pack(fill='x', padx=2)
        
        self.hash_model_status = tk.Label(hash_model_frame, text="⚠ Hash model not loaded",
                                          bg=COLORS['bg_medium'], fg=COLORS['warning'],
                                          font=('Arial', 8, 'bold'))
        self.hash_model_status.grid(row=3, column=0, columnspan=3, pady=2)
        
        # Hash input section
        input_frame = ttk.LabelFrame(parent, text="📝 Hash Input", padding=10)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(input_frame, text="Enter Hash(es) - one per line:", 
                font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).pack(anchor='w', pady=2)
        
        self.hash_input = scrolledtext.ScrolledText(input_frame, height=5,
                                                    font=('Courier', 9),
                                                    bg='white', fg='black',
                                                    wrap=tk.WORD)
        self.hash_input.pack(fill='x', pady=5)
        
        # Quick test buttons
        quick_frame = tk.Frame(input_frame, bg=COLORS['bg_medium'])
        quick_frame.pack(fill='x', pady=5)
        
        tk.Label(quick_frame, text="Quick Test:", font=('Arial', 8),
                bg=COLORS['bg_medium'], fg=COLORS['text_gray']).pack(side='left', padx=2)
        
        tk.Button(quick_frame, text="MD5", command=lambda: self.insert_test_hash('MD5'),
                 bg=COLORS['info'], fg='white', font=('Arial', 7, 'bold')).pack(side='left', padx=1)
        tk.Button(quick_frame, text="SHA1", command=lambda: self.insert_test_hash('SHA1'),
                 bg=COLORS['info'], fg='white', font=('Arial', 7, 'bold')).pack(side='left', padx=1)
        tk.Button(quick_frame, text="SHA256", command=lambda: self.insert_test_hash('SHA256'),
                 bg=COLORS['info'], fg='white', font=('Arial', 7, 'bold')).pack(side='left', padx=1)
        tk.Button(quick_frame, text="NTLM", command=lambda: self.insert_test_hash('NTLM'),
                 bg=COLORS['info'], fg='white', font=('Arial', 7, 'bold')).pack(side='left', padx=1)
        
        # Action buttons
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_medium'])
        btn_frame.pack(fill='x', pady=5)
        
        tk.Button(btn_frame, text="🔍 Identify Hash", command=self.identify_hashes,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 10, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=2)
        
        tk.Button(btn_frame, text="🧹 Clear", command=self.clear_hash_results,
                 bg=COLORS['accent_red'], fg='white', font=('Arial', 10, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=2)
        
        # Hash results table
        results_frame = ttk.LabelFrame(parent, text="📊 Identification Results", padding=5)
        results_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Scrollbars
        hash_scroll_y = ttk.Scrollbar(results_frame, orient='vertical')
        hash_scroll_y.pack(side='right', fill='y')
        
        # Treeview
        hash_columns = ('Hash', 'Type', 'Confidence', 'Probability')
        self.hash_results_tree = ttk.Treeview(results_frame, columns=hash_columns, 
                                             show='headings',
                                             style='Custom.Treeview',
                                             yscrollcommand=hash_scroll_y.set,
                                             height=6)
        
        hash_scroll_y.config(command=self.hash_results_tree.yview)
        
        # Configure columns
        self.hash_results_tree.heading('Hash', text='Hash')
        self.hash_results_tree.heading('Type', text='Type')
        self.hash_results_tree.heading('Confidence', text='Confidence')
        self.hash_results_tree.heading('Probability', text='Prob%')
        
        self.hash_results_tree.column('Hash', width=150)
        self.hash_results_tree.column('Type', width=100, anchor='center')
        self.hash_results_tree.column('Confidence', width=70, anchor='center')
        self.hash_results_tree.column('Probability', width=60, anchor='center')
        
        self.hash_results_tree.pack(fill='both', expand=True)
        
        # Color tags for hash types
        self.hash_results_tree.tag_configure('high', background='#27ae60', foreground='white')
        self.hash_results_tree.tag_configure('medium', background='#f39c12', foreground='white')
        self.hash_results_tree.tag_configure('low', background='#e74c3c', foreground='white')
        
        # Hash details
        detail_frame = ttk.LabelFrame(parent, text="📋 Hash Details & Recommendations", padding=5)
        detail_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.hash_detail_text = scrolledtext.ScrolledText(detail_frame, height=6,
                                                          bg=COLORS['bg_dark'],
                                                          fg=COLORS['accent_blue'],
                                                          font=('Courier', 8))
        self.hash_detail_text.pack(fill='both', expand=True)
        
        # Bind selection
        self.hash_results_tree.bind('<<TreeviewSelect>>', self.on_hash_select)
        
        # Statistics panel
        stats_frame = tk.Frame(parent, bg=COLORS['bg_light'], relief='raised', bd=2)
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(stats_frame, text="📊 Statistics", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_light'], fg=COLORS['accent_cyan']).pack(pady=3)
        
        self.hash_stats_label = tk.Label(stats_frame, 
                                         text="Total: 0 | Identified: 0 | Unknown: 0",
                                         font=('Arial', 8),
                                         bg=COLORS['bg_light'], 
                                         fg=COLORS['text_white'])
        self.hash_stats_label.pack(pady=2)
    
    def insert_test_hash(self, hash_type):
        """Insert test hash for quick testing"""
        test_hashes = {
            'MD5': '5f4dcc3b5aa765d61d8327deb882cf99',  # password
            'SHA1': '5baa61e4c9b93f3f0682250b6cf8331b7ee68fd8',  # password
            'SHA256': '5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8',  # password
            'NTLM': '8846f7eaee8fb117ad06bdd830b7586c'  # password
        }
        
        current_text = self.hash_input.get(1.0, tk.END).strip()
        if current_text:
            self.hash_input.insert(tk.END, '\n' + test_hashes[hash_type])
        else:
            self.hash_input.insert(1.0, test_hashes[hash_type])
    
    def identify_hashes(self):
        """Identify hash types using ML-enhanced identification"""
        hashes = self.hash_input.get(1.0, tk.END).strip().split('\n')
        hashes = [h.strip() for h in hashes if h.strip()]
        
        if not hashes:
            messagebox.showerror("Error", "Please enter at least one hash")
            return
        
        # Clear previous results
        for item in self.hash_results_tree.get_children():
            self.hash_results_tree.delete(item)
        
        self.hash_detail_text.delete(1.0, tk.END)
        
        # Store results
        self.hash_identification_results = {}
        identified_count = 0
        unknown_count = 0
        
        for hash_value in hashes:
            # Use ML identifier
            result = self.ml_hash_identifier.identify(hash_value)
            self.hash_identification_results[hash_value] = result
            
            # Determine tag based on confidence
            confidence = result['confidence']
            if confidence == 'High':
                tag = 'high'
                identified_count += 1
            elif confidence == 'Medium':
                tag = 'medium'
                identified_count += 1
            else:
                tag = 'low'
                unknown_count += 1
            
            # Mask hash for display
            masked_hash = hash_value[:16] + "..." if len(hash_value) > 16 else hash_value
            
            # Format probability
            prob_percent = f"{result['probability']*100:.1f}%"
            
            # Insert into tree
            self.hash_results_tree.insert('', 'end',
                                         values=(
                                             masked_hash,
                                             result['type'],
                                             confidence,
                                             prob_percent
                                         ),
                                         tags=(tag,))
        
        # Update statistics
        self.hash_stats_label.config(
            text=f"Total: {len(hashes)} | Identified: {identified_count} | Unknown: {unknown_count}"
        )
        
        # Show if ML was used
        ml_status = "with ML" if self.ml_hash_identifier.model_loaded else "rule-based"
        self.status_label.config(text=f"✓ Identified {len(hashes)} hash(es) ({ml_status})",
                                fg=COLORS['success'])
    
    def on_hash_select(self, event):
        """Display detailed hash information"""
        selection = self.hash_results_tree.selection()
        if not selection:
            return
        
        item = self.hash_results_tree.item(selection[0])
        hash_masked = item['values'][0]
        
        # Find full hash
        for hash_val, result in self.hash_identification_results.items():
            if hash_val.startswith(hash_masked.replace('...', '')):
                self.display_hash_details(hash_val, result)
                break
    
    def display_hash_details(self, hash_value, result):
        """Display detailed hash analysis"""
        self.hash_detail_text.delete(1.0, tk.END)
        
        info = f"{'='*60}\n"
        info += f"HASH ANALYSIS\n"
        info += f"{'='*60}\n\n"
        info += f"Hash: {hash_value}\n"
        info += f"Length: {len(hash_value)} characters\n\n"
        
        info += f"IDENTIFICATION:\n"
        info += f"  Type: {result['type']}\n"
        info += f"  Confidence: {result['confidence']}\n"
        info += f"  Probability: {result['probability']*100:.1f}%\n"
        info += f"  Method: {'ML Model' if result.get('ml_prediction', False) else 'Rule-Based'}\n\n"
        
        if result.get('alternatives') and len(result['alternatives']) > 0:
            info += f"  Alternative Types:\n"
            for alt in result['alternatives']:
                info += f"    - {alt['type']} ({alt['probability']*100:.1f}%)\n"
            info += "\n"
        
        if 'recommendation' in result:
            info += f"RECOMMENDATION:\n"
            info += f"  {result['recommendation']}\n\n"
        
        info += f"{'='*60}\n"
        
        # Add cracking recommendations
        info += f"CRACKING STRATEGY:\n"
        
        hash_type = result['type']
        
        if hash_type in ['MD5', 'SHA1', 'NTLM']:
            info += "  ✓ Fast hash - Dictionary + Brute force viable\n"
            info += "  ✓ GPU acceleration recommended\n"
            info += "  ✓ Wordlist: Start with common passwords\n"
            info += "  ✓ Rules: Apply leetspeak, years, symbols\n"
            info += "  ✓ Expected time: Minutes to hours (GPU)\n"
        elif hash_type in ['SHA256', 'SHA512', 'SHA224', 'SHA384']:
            info += "  ✓ Medium speed - Dictionary preferred\n"
            info += "  ✓ GPU can help significantly\n"
            info += "  ✓ Targeted wordlist recommended\n"
            info += "  ⚠ Brute force: Only for short passwords\n"
            info += "  ✓ Expected time: Hours to days (GPU)\n"
        elif 'Bcrypt' in hash_type or 'ARGON' in hash_type.upper():
            info += "  ⚠ Very slow hash - GPU strongly recommended\n"
            info += "  ⚠ Targeted dictionary attack only\n"
            info += "  ⚠ Brute force not practical\n"
            info += "  ⚠ Expected time: Days to weeks (even with GPU)\n"
        elif hash_type == 'MySQL_OLD':
            info += "  ✓ Very weak hash - Extremely fast cracking\n"
            info += "  ✓ Dictionary attack: Seconds to minutes\n"
            info += "  ✓ Brute force viable for short passwords\n"
        elif hash_type in ['WordPress', 'Joomla']:
            info += "  ✓ Salted hash - Slower than plain MD5\n"
            info += "  ✓ Dictionary attack recommended\n"
            info += "  ✓ GPU acceleration helpful\n"
            info += "  ✓ Expected time: Hours to days\n"
        elif hash_type == 'Base64_Encoded':
            info += "  ⚠ Base64 encoding detected\n"
            info += "  ⚠ Decode first to identify actual hash\n"
            info += "  ⚠ May contain binary data\n"
        elif hash_type == 'Unknown' or 'Unknown' in hash_type:
            info += "  ⚠ Unknown type - Manual analysis required\n"
            info += "  • Check hash length\n"
            info += "  • Look for special characters ($, :)\n"
            info += "  • Consult hash identification tools\n"
            info += "  • Try online hash databases\n"
        
        info += f"\n{'='*60}\n"
        
        # Add hash format info
        info += f"FORMAT DETAILS:\n"
        if re.match(r'^[a-fA-F0-9]+$', hash_value):
            info += "  • Character Set: Hexadecimal (0-9, a-f)\n"
            info += "  • Encoding: Standard hex representation\n"
        elif '$' in hash_value:
            info += "  • Format: Modular Crypt Format\n"
            info += "  • Contains: Salt and parameters\n"
            if hash_value.startswith('$2'):
                info += "  • Type: Bcrypt family\n"
            elif hash_value.startswith('$P$') or hash_value.startswith('$H$'):
                info += "  • Type: Portable PHP hash\n"
        elif ':' in hash_value:
            info += "  • Format: Salted or Network Auth\n"
            info += "  • May contain: Username, domain, salt\n"
        elif '/' in hash_value or '+' in hash_value or '=' in hash_value:
            info += "  • Format: Possibly Base64 encoded\n"
            info += "  • Action: Try decoding first\n"
        
        self.hash_detail_text.insert(1.0, info)
    
    def clear_hash_results(self):
        """Clear hash identification results"""
        for item in self.hash_results_tree.get_children():
            self.hash_results_tree.delete(item)
        self.hash_detail_text.delete(1.0, tk.END)
        self.hash_input.delete(1.0, tk.END)
        self.hash_stats_label.config(text="Total: 0 | Identified: 0 | Unknown: 0")
        self.status_label.config(text="✓ Cleared", fg=COLORS['success'])
    
    def browse_model_file(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Password Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.ml_model_path.delete(0, tk.END)
            self.ml_model_path.insert(0, filename)
    
    def browse_features_file(self):
        """Browse for features file"""
        filename = filedialog.askopenfilename(
            title="Select Password Features File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.ml_features_path.delete(0, tk.END)
            self.ml_features_path.insert(0, filename)
    
    def browse_hash_model_file(self):
        """Browse for hash model file"""
        filename = filedialog.askopenfilename(
            title="Select Hash Model File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if filename:
            self.hash_model_path.delete(0, tk.END)
            self.hash_model_path.insert(0, filename)
    
    def browse_hash_features_file(self):
        """Browse for hash features file"""
        filename = filedialog.askopenfilename(
            title="Select Hash Features File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.hash_features_path.delete(0, tk.END)
            self.hash_features_path.insert(0, filename)
    
    def load_ml_model(self):
        """Load password ML model"""
        if not ML_AVAILABLE:
            messagebox.showerror("Error", "ML libraries not available. Install pandas and joblib.")
            return
        
        model_path = self.ml_model_path.get().strip()
        features_path = self.ml_features_path.get().strip()
        
        if not model_path or not features_path:
            messagebox.showerror("Error", "Please specify both model and features files")
            return
        
        try:
            self.ml_analyzer = MLPasswordAnalyzer(model_path, features_path)
            if self.ml_analyzer.model_loaded:
                self.ml_model_status.config(text="✓ Password model loaded",
                                          fg=COLORS['success'])
                messagebox.showinfo("Success", "Password ML model loaded successfully!")
            else:
                self.ml_model_status.config(text="⚠ Model loading failed",
                                          fg=COLORS['danger'])
                messagebox.showerror("Error", "Failed to load password model files")
        except Exception as e:
            self.ml_model_status.config(text="⚠ Model loading error",
                                      fg=COLORS['danger'])
            messagebox.showerror("Error", f"Password model loading error: {str(e)}")
    
    def load_hash_ml_model(self):
        """Load hash ML model"""
        if not ML_AVAILABLE:
            messagebox.showerror("Error", "ML libraries not available. Install pandas, joblib and scikit-learn.")
            return
        
        model_path = self.hash_model_path.get().strip()
        features_path = self.hash_features_path.get().strip()
        
        if not model_path or not features_path:
            messagebox.showerror("Error", "Please specify both model and features files")
            return
        
        try:
            self.ml_hash_identifier = MLHashIdentifier(model_path, features_path)
            if self.ml_hash_identifier.model_loaded:
                self.hash_model_status.config(text="✓ Hash model loaded",
                                             fg=COLORS['success'])
                messagebox.showinfo("Success", "Hash ML model loaded successfully!")
            else:
                self.hash_model_status.config(text="⚠ Model loading failed",
                                             fg=COLORS['danger'])
                messagebox.showerror("Error", "Failed to load hash model files")
        except Exception as e:
            self.hash_model_status.config(text="⚠ Model loading error",
                                         fg=COLORS['danger'])
            messagebox.showerror("Error", f"Hash model loading error: {str(e)}")
    
    def analyze_passwords_ml(self):
        """Analyze passwords with ML"""
        passwords = self.ml_password_input.get(1.0, tk.END).strip().split('\n')
        passwords = [p.strip() for p in passwords if p.strip()]
        
        if not passwords:
            messagebox.showerror("Error", "Please enter at least one password")
            return
        
        # Clear previous results
        for item in self.ml_results_tree.get_children():
            self.ml_results_tree.delete(item)
        
        self.ml_detail_text.delete(1.0, tk.END)
        
        # Analyze each password
        self.password_analysis_results = {}
        
        for password in passwords:
            result = self.ml_analyzer.analyze(password)
            self.password_analysis_results[password] = result
            
            # Determine tag for row color
            prediction = result.get('ml_prediction', result['rule_based_label'])
            tag = prediction.lower()
            
            # Get confidence
            if 'ml_confidence' in result and result['ml_confidence']:
                conf = result['ml_confidence'].get(prediction, 0) * 100
                confidence_str = f"{conf:.1f}%"
            else:
                confidence_str = "N/A"
            
            # Mask password for display
            masked_pw = password if len(password) <= 20 else password[:17] + "..."
            
            # Insert into tree
            self.ml_results_tree.insert('', 'end',
                                       values=(
                                           masked_pw,
                                           result['length'],
                                           f"{result['entropy_bits']:.2f}",
                                           f"{result['strength_score']:.2f}",
                                           result['rule_based_label'],
                                           result.get('ml_prediction', 'N/A'),
                                           confidence_str
                                       ),
                                       tags=(tag,))
        
        self.status_label.config(text=f"✓ Analyzed {len(passwords)} password(s)",
                                fg=COLORS['success'])
    
    def on_password_select(self, event):
        """Display detailed info when password is selected"""
        selection = self.ml_results_tree.selection()
        if not selection:
            return
        
        item = self.ml_results_tree.item(selection[0])
        password = item['values'][0]
        
        # Find full password in results
        for pw, result in self.password_analysis_results.items():
            if pw.startswith(password) or password.startswith(pw[:17]):
                self.display_detailed_info(pw, result)
                break
    
    def display_detailed_info(self, password, result):
        """Display detailed password analysis"""
        self.ml_detail_text.delete(1.0, tk.END)
        
        info = f"{'='*70}\n"
        info += f"DETAILED ANALYSIS\n"
        info += f"{'='*70}\n\n"
        info += f"Password: {password}\n\n"
        
        info += f"BASIC METRICS:\n"
        info += f"  Length: {result['length']} characters\n"
        info += f"  Entropy: {result['entropy_bits']:.2f} bits\n"
        info += f"  Unique Characters: {result['unique_ratio']*100:.1f}%\n"
        info += f"  Longest Run: {result['longest_run']} characters\n\n"
        
        info += f"CHARACTER COMPOSITION:\n"
        info += f"  Lowercase: {'✓' if result['has_lower'] else '✗'}\n"
        info += f"  Uppercase: {'✓' if result['has_upper'] else '✗'}\n"
        info += f"  Digits: {'✓' if result['has_digit'] else '✗'}\n"
        info += f"  Symbols: {'✓' if result['has_symbol'] else '✗'}\n\n"
        
        info += f"PATTERN DETECTION:\n"
        info += f"  Common Word: {'⚠ YES' if result['has_common_word'] else '✓ NO'}\n"
        info += f"  Year Pattern: {'⚠ YES' if result['has_year_like'] else '✓ NO'}\n"
        info += f"  Keyboard Walk: {'⚠ YES' if result['has_keyboard_walk'] else '✓ NO'}\n"
        info += f"  Sequential Digits: {'⚠ YES' if result['has_sequential_digits'] else '✓ NO'}\n\n"
        
        info += f"STRENGTH ASSESSMENT:\n"
        info += f"  Strength Score: {result['strength_score']:.2f}\n"
        info += f"  Rule-Based: {result['rule_based_label']}\n"
        
        if 'ml_prediction' in result and result['ml_prediction'] != 'N/A':
            info += f"  ML Prediction: {result['ml_prediction']}\n\n"
            
            if 'ml_confidence' in result:
                info += f"  ML CONFIDENCE:\n"
                for label, conf in result['ml_confidence'].items():
                    info += f"    {label}: {conf*100:.1f}%\n"
        else:
            info += f"  ML Prediction: Not available (load model first)\n"
        
        info += f"\n{'='*70}\n"
        
        # Add recommendations
        info += f"RECOMMENDATIONS:\n"
        recommendations = []
        
        if result['length'] < 12:
            recommendations.append("• Increase length to at least 12 characters")
        if not result['has_upper']:
            recommendations.append("• Add uppercase letters")
        if not result['has_digit']:
            recommendations.append("• Add numbers")
        if not result['has_symbol']:
            recommendations.append("• Add special characters")
        if result['has_common_word']:
            recommendations.append("• Avoid common words")
        if result['has_year_like']:
            recommendations.append("• Remove year patterns")
        if result['unique_ratio'] < 0.7:
            recommendations.append("• Use more unique characters")
        if result['longest_run'] >= 3:
            recommendations.append("• Avoid repeated characters")
        
        if recommendations:
            for rec in recommendations:
                info += f"{rec}\n"
        else:
            info += "✓ Password meets best practices!\n"
        
        self.ml_detail_text.insert(1.0, info)
    
    def generate_test_dataset(self):
        """Generate test password dataset"""
        messagebox.showinfo("Generate Dataset",
                          "This will generate 100 test passwords.\nCheck the ML Password Analyzer tab.")
        
        test_passwords = []
        
        # Weak passwords
        weak = ["password", "123456", "qwerty", "abc123", "Password1", 
                "admin123", "welcome1", "iloveyou"]
        test_passwords.extend(weak)
        
        # Medium passwords
        for i in range(20):
            length = random.randint(8, 12)
            pw = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            test_passwords.append(pw)
        
        # Strong passwords
        for i in range(20):
            length = random.randint(14, 20)
            pw = ''.join(random.choices(
                string.ascii_letters + string.digits + "!@#$%^&*()_+-=", k=length))
            test_passwords.append(pw)
        
        # Pattern-based passwords
        patterns = ["ahmed2024!", "Sara@1990", "Omar#12345", "maya_test123"]
        test_passwords.extend(patterns)
        
        # Random samples to reach 100
        while len(test_passwords) < 100:
            length = random.randint(6, 16)
            pw = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            test_passwords.append(pw)
        
        # Insert into text box
        self.ml_password_input.delete(1.0, tk.END)
        self.ml_password_input.insert(1.0, '\n'.join(test_passwords[:50]))  # First 50
        
        messagebox.showinfo("Dataset Generated", 
                          f"Generated {len(test_passwords[:50])} test passwords.\n"
                          "Click 'Analyze' to process them.")
    
    def clear_ml_results(self):
        """Clear ML results"""
        for item in self.ml_results_tree.get_children():
            self.ml_results_tree.delete(item)
        self.ml_detail_text.delete(1.0, tk.END)
        self.ml_password_input.delete(1.0, tk.END)
        self.status_label.config(text="✓ Cleared", fg=COLORS['success'])
    
    # =========================================================================
    # HASH CRACKER TAB (Simplified - keeping original functionality)
    # =========================================================================
    
    def create_hash_cracker_tab(self):
        """Create hash cracker tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔓 Hash Cracker")
        
        # Header
        header = tk.Frame(tab, bg=COLORS['accent_green'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🔓 Password Hash Cracker",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_green'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Input section
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Hash:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.hash_entry = tk.Entry(input_frame, width=60, font=('Arial', 10))
        self.hash_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5)
        
        tk.Label(input_frame, text="Hash Type:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.hash_type = ttk.Combobox(input_frame, values=['MD5', 'SHA1', 'SHA256', 'SHA512', 'NTLM'],
                                     state='readonly', width=20)
        self.hash_type.grid(row=1, column=1, sticky='w', padx=5, pady=5)
        self.hash_type.current(0)
        
        tk.Button(input_frame, text="🔍 Identify Hash", command=self.identify_hash_type,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 9, 'bold')).grid(row=1, column=2, padx=5)
        
        # Attack options
        attack_frame = ttk.LabelFrame(tab, text="Attack Options", padding=10)
        attack_frame.pack(fill='x', padx=10, pady=5)
        
        # Wordlist section
        self.use_wordlist_var = tk.BooleanVar(value=False)
        tk.Checkbutton(attack_frame, text="Use Wordlist", variable=self.use_wordlist_var,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                      selectcolor=COLORS['bg_dark'], font=('Arial', 10)).grid(row=0, column=0, padx=5, sticky='w')
        
        tk.Button(attack_frame, text="📁 Upload Wordlist", command=self.upload_wordlist,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 9, 'bold')).grid(row=0, column=1, padx=5)
        
        self.wordlist_label = tk.Label(attack_frame, text="No wordlist loaded", 
                                       bg=COLORS['bg_medium'], fg=COLORS['accent_cyan'],
                                       font=('Arial', 9))
        self.wordlist_label.grid(row=0, column=2, padx=5, sticky='w')
        
        # Brute-force section
        self.use_brute_var = tk.BooleanVar(value=True)
        tk.Checkbutton(attack_frame, text="Use Brute Force", variable=self.use_brute_var,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                      selectcolor=COLORS['bg_dark'], font=('Arial', 10)).grid(row=1, column=0, padx=5, sticky='w')
        
        tk.Label(attack_frame, text="Min Length:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=1, sticky='w', padx=5)
        self.brute_min_length = tk.Spinbox(attack_frame, from_=1, to=10, width=5, font=('Arial', 10))
        self.brute_min_length.insert(0, '1')
        self.brute_min_length.grid(row=1, column=2, padx=5, sticky='w')
        
        tk.Label(attack_frame, text="Max Length:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=2, column=1, sticky='w', padx=5)
        self.brute_max_length = tk.Spinbox(attack_frame, from_=1, to=10, width=5, font=('Arial', 10))
        self.brute_max_length.insert(0, '4')
        self.brute_max_length.grid(row=2, column=2, padx=5, sticky='w')
        
        # Store wordlist data
        self.loaded_wordlist = None
        
        # Control buttons
        btn_frame = tk.Frame(tab, bg=COLORS['bg_medium'])
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="▶ Start Attack", command=self.start_crack,
                 bg=COLORS['success'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        tk.Button(btn_frame, text="⏸ Pause", command=self.pause_attack,
                 bg=COLORS['warning'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        tk.Button(btn_frame, text="⏹ Stop", command=self.stop_attack,
                 bg=COLORS['danger'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.crack_output = scrolledtext.ScrolledText(output_frame, height=20,
                                                      bg=COLORS['bg_dark'],
                                                      fg=COLORS['accent_green'],
                                                      font=('Courier', 10))
        self.crack_output.pack(fill='both', expand=True)
    
    
    def upload_wordlist(self):
        """Upload a wordlist file"""
        file_path = filedialog.askopenfilename(
            title="Select Wordlist File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self.loaded_wordlist = [line.strip() for line in f if line.strip()]
                
                word_count = len(self.loaded_wordlist)
                filename = file_path.split('/')[-1]
                self.wordlist_label.config(text=f"✓ {filename} ({word_count} words)")
                self.crack_output.insert(tk.END, f"\n✓ Loaded wordlist: {filename} with {word_count} words\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load wordlist: {e}")
                self.wordlist_label.config(text="Error loading wordlist")
    
    def identify_hash_type(self):
        """Identify hash type"""
        hash_value = self.hash_entry.get().strip()
        if not hash_value:
            messagebox.showerror("Error", "Please enter a hash")
            return
        
        hash_type, recommendation = identify_hash(hash_value)
        self.crack_output.insert(tk.END, f"\n{'='*70}\n")
        self.crack_output.insert(tk.END, f"Hash Type: {hash_type}\n")
        self.crack_output.insert(tk.END, f"Recommendation: {recommendation}\n")
        self.crack_output.insert(tk.END, f"{'='*70}\n")
        
        # Try to set the combo box
        if hash_type in ['MD5', 'SHA1', 'SHA256', 'SHA512', 'NTLM']:
            idx = ['MD5', 'SHA1', 'SHA256', 'SHA512', 'NTLM'].index(hash_type)
            self.hash_type.current(idx)
    
    def start_crack(self):
        """Start cracking"""
        hash_value = self.hash_entry.get().strip()
        hash_type = self.hash_type.get()
        
        if not hash_value or not hash_type:
            messagebox.showerror("Error", "Please enter hash and select type")
            return
        
        SETTINGS.attack_stopped = False
        SETTINGS.attack_paused = False
        
        self.crack_output.insert(tk.END, f"\n{'='*70}\n")
        self.crack_output.insert(tk.END, f"Starting {hash_type} crack...\n")
        self.crack_output.insert(tk.END, f"Target: {hash_value}\n")
        self.crack_output.insert(tk.END, f"{'='*70}\n")
        
        # Start cracking thread
        thread = threading.Thread(target=self._crack_thread, args=(hash_value, hash_type), daemon=True)
        thread.start()
    
    def _crack_thread(self, hash_value, hash_type):
        """Cracking thread"""
        wordlist = None
        
        # Use uploaded wordlist if available, otherwise use common passwords
        if self.use_wordlist_var.get():
            if self.loaded_wordlist:
                wordlist = self.loaded_wordlist
            else:
                wordlist = COMMON_PASSWORDS
        
        def progress_cb(attempts, current):
            if attempts % 1000 == 0:
                self.master.after(0, lambda: self.crack_output.insert(tk.END, 
                                  f"Attempts: {attempts:,} | Testing: {current}\n"))
        
        # Get brute-force length options
        min_length = int(self.brute_min_length.get())
        max_length = int(self.brute_max_length.get())
        
        result = PasswordCracker.crack_hash(
            hash_value, 
            hash_type,
            wordlist=wordlist,
            use_combinations=self.use_brute_var.get(),
            combo_min=min_length,
            combo_max=max_length,
            max_attempts=100000,
            progress_callback=progress_cb
        )
        
        self.master.after(0, lambda: self._display_crack_result(result))
    
    def _display_crack_result(self, result):
        """Display crack result"""
        self.crack_output.insert(tk.END, f"\n{'='*70}\n")
        if result['success']:
            self.crack_output.insert(tk.END, f"✓ PASSWORD FOUND: {result['password']}\n")
            self.crack_output.insert(tk.END, f"Attempts: {result['attempts']:,}\n")
            self.crack_output.insert(tk.END, f"Time: {result['time']}s\n")
            self.crack_output.insert(tk.END, f"Rate: {result['rate']:,} H/s\n")
        else:
            self.crack_output.insert(tk.END, f"✗ Password not found\n")
            self.crack_output.insert(tk.END, f"Attempts: {result.get('attempts', 0):,}\n")
        self.crack_output.insert(tk.END, f"{'='*70}\n")
    
    def pause_attack(self):
        """Pause attack"""
        SETTINGS.attack_paused = not SETTINGS.attack_paused
        status = "Paused" if SETTINGS.attack_paused else "Resumed"
        self.crack_output.insert(tk.END, f"⏸ {status}\n")
    
    def stop_attack(self):
        """Stop attack"""
        SETTINGS.attack_stopped = True
        self.crack_output.insert(tk.END, "⏹ Attack stopped\n")
    
    # =========================================================================
    # OTHER TABS (Simplified versions - keeping core functionality)
    # =========================================================================
    
    def create_network_tools_tab(self):
        """Network tools tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🌐 Network Tools")
        
        header = tk.Frame(tab, bg=COLORS['accent_cyan'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🌐 Network Diagnostic Tools",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_cyan'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Target:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.net_target = tk.Entry(input_frame, width=40, font=('Arial', 10))
        self.net_target.grid(row=0, column=1, padx=5)
        self.net_target.insert(0, "google.com")
        
        # Buttons
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_medium'])
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        tk.Button(btn_frame, text="📡 Ping", command=self.do_ping,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(btn_frame, text="🛤 Traceroute", command=self.do_traceroute,
                 bg=COLORS['accent_purple'], fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.net_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                    bg=COLORS['bg_dark'],
                                                    fg=COLORS['accent_cyan'],
                                                    font=('Courier', 9))
        self.net_output.pack(fill='both', expand=True)
    
    def do_ping(self):
        """Ping target"""
        target = self.net_target.get().strip()
        if not target:
            messagebox.showerror("Error", "Enter target")
            return
        
        self.net_output.insert(tk.END, f"\nPinging {target}...\n")
        threading.Thread(target=self._ping_thread, args=(target,), daemon=True).start()
    
    def _ping_thread(self, target):
        """Ping thread"""
        result = NetworkTools.ping(target)
        if result['success']:
            self.master.after(0, lambda: self.net_output.insert(tk.END, result['output'] + "\n"))
        else:
            self.master.after(0, lambda: self.net_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def do_traceroute(self):
        """Traceroute"""
        target = self.net_target.get().strip()
        if not target:
            messagebox.showerror("Error", "Enter target")
            return
        
        self.net_output.insert(tk.END, f"\nTraceroute to {target}...\n")
        threading.Thread(target=self._traceroute_thread, args=(target,), daemon=True).start()
    
    def _traceroute_thread(self, target):
        """Traceroute thread"""
        result = NetworkTools.traceroute(target)
        if result['success']:
            self.master.after(0, lambda: self.net_output.insert(tk.END, result['output'] + "\n"))
        else:
            self.master.after(0, lambda: self.net_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def create_online_attacks_tab(self):
        """Online attacks tab for SSH and port brute-force"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔓 Online Attacks")
        
        header = tk.Frame(tab, bg=COLORS['accent_red'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🔓 Online Brute-Force Attacks",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_red'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Configuration
        config_frame = ttk.LabelFrame(tab, text="Attack Configuration", padding=15)
        config_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(config_frame, text="Attack Type:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.attack_type = ttk.Combobox(config_frame, values=['SSH', 'Port'], state='readonly', width=20)
        self.attack_type.grid(row=0, column=1, padx=5, pady=5)
        self.attack_type.current(0)
        
        tk.Label(config_frame, text="Target Host:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.attack_target = tk.Entry(config_frame, width=40, font=('Arial', 10))
        self.attack_target.grid(row=1, column=1, padx=5, pady=5)
        self.attack_target.insert(0, "192.168.1.1")
        
        tk.Label(config_frame, text="Port:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.attack_port = tk.Entry(config_frame, width=40, font=('Arial', 10))
        self.attack_port.grid(row=2, column=1, padx=5, pady=5)
        self.attack_port.insert(0, "22")
        
        tk.Label(config_frame, text="Username:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.attack_username = tk.Entry(config_frame, width=40, font=('Arial', 10))
        self.attack_username.grid(row=3, column=1, padx=5, pady=5)
        self.attack_username.insert(0, "root")
        
        tk.Label(config_frame, text="Wordlist:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=4, column=0, sticky='w', padx=5, pady=5)
        tk.Button(config_frame, text="📁 Load Wordlist", command=self.load_attack_wordlist,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 9, 'bold')).grid(row=4, column=1, sticky='w', padx=5)
        
        self.attack_wordlist_label = tk.Label(config_frame, text="No wordlist loaded",
                                             bg=COLORS['bg_medium'], fg=COLORS['accent_cyan'],
                                             font=('Arial', 9))
        self.attack_wordlist_label.grid(row=4, column=2, padx=5, sticky='w')
        
        # Control buttons
        btn_frame = tk.Frame(tab, bg=COLORS['bg_medium'])
        btn_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(btn_frame, text="▶ Start Attack", command=self.start_online_attack,
                 bg=COLORS['success'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        tk.Button(btn_frame, text="⏸ Pause", command=self.pause_online_attack,
                 bg=COLORS['warning'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        tk.Button(btn_frame, text="⏹ Stop", command=self.stop_online_attack,
                 bg=COLORS['danger'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).pack(side='left', fill='x', expand=True, padx=5)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.attack_output = scrolledtext.ScrolledText(output_frame, height=20,
                                                       bg=COLORS['bg_dark'],
                                                       fg=COLORS['accent_orange'],
                                                       font=('Courier', 10))
        self.attack_output.pack(fill='both', expand=True)
        
        # Store attack data
        self.attack_wordlist = None
        self.attack_stopped = False
        self.attack_paused = False
    
    def load_attack_wordlist(self):
        """Load wordlist for online attacks"""
        file_path = filedialog.askopenfilename(
            title="Select Wordlist File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self.attack_wordlist = [line.strip() for line in f if line.strip()]
                
                word_count = len(self.attack_wordlist)
                filename = file_path.split('/')[-1]
                self.attack_wordlist_label.config(text=f"✓ {filename} ({word_count} words)")
                self.attack_output.insert(tk.END, f"\n✓ Loaded wordlist: {filename} with {word_count} words\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load wordlist: {e}")
    
    def start_online_attack(self):
        """Start online brute-force attack"""
        if not self.attack_wordlist:
            messagebox.showerror("Error", "Please load a wordlist first")
            return
        
        attack_type = self.attack_type.get()
        target = self.attack_target.get().strip()
        port = self.attack_port.get().strip()
        username = self.attack_username.get().strip()
        
        if not target or not port:
            messagebox.showerror("Error", "Please enter target and port")
            return
        
        self.attack_stopped = False
        self.attack_paused = False
        
        self.attack_output.insert(tk.END, f"\n{'='*70}\n")
        self.attack_output.insert(tk.END, f"Starting {attack_type} brute-force attack...\n")
        self.attack_output.insert(tk.END, f"Target: {target}:{port}\n")
        self.attack_output.insert(tk.END, f"Username: {username}\n")
        self.attack_output.insert(tk.END, f"Wordlist: {len(self.attack_wordlist)} passwords\n")
        self.attack_output.insert(tk.END, f"{'='*70}\n")
        
        threading.Thread(target=self._online_attack_thread, args=(attack_type, target, port, username), daemon=True).start()
    
    def _online_attack_thread(self, attack_type, target, port, username):
        """Online attack thread"""
        attempts = 0
        found = False
        
        for password in self.attack_wordlist:
            if self.attack_stopped:
                self.master.after(0, lambda: self.attack_output.insert(tk.END, "\nAttack stopped by user\n"))
                break
            
            while self.attack_paused:
                time.sleep(0.1)
            
            attempts += 1
            
            if attempts % 10 == 0:
                self.master.after(0, lambda a=attempts, p=password: self.attack_output.insert(tk.END, f"Attempt {a}: {p}\n"))
        
        if not found:
            self.master.after(0, lambda: self.attack_output.insert(tk.END, f"\nAttack completed. {attempts} attempts made.\n"))
    
    def pause_online_attack(self):
        """Pause online attack"""
        self.attack_paused = not self.attack_paused
        status = "Paused" if self.attack_paused else "Resumed"
        self.attack_output.insert(tk.END, f"\n{status}\n")
    
    def stop_online_attack(self):
        """Stop online attack"""
        self.attack_stopped = True
        self.attack_output.insert(tk.END, "\nStopping attack...\n")
    
    def create_vulnerability_scanner_tab(self):
        """Vulnerability scanner tab for SQL injection and XSS"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Vuln Scanner")
        
        header = tk.Frame(tab, bg=COLORS['accent_red'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🔍 Vulnerability Scanner",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_red'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Target URL:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.vuln_url = tk.Entry(input_frame, width=60, font=('Arial', 10))
        self.vuln_url.grid(row=0, column=1, padx=5)
        self.vuln_url.insert(0, "http://example.com")
        
        btn_frame = tk.Frame(input_frame, bg=COLORS['bg_medium'])
        btn_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        tk.Button(btn_frame, text="🔍 Scan SQLi", command=self.scan_sqli,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(btn_frame, text="🔍 Scan XSS", command=self.scan_xss,
                 bg=COLORS['accent_blue'], fg='white', font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.vuln_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                     bg=COLORS['bg_dark'],
                                                     fg=COLORS['accent_red'],
                                                     font=('Courier', 9))
        self.vuln_output.pack(fill='both', expand=True)
    
    def scan_sqli(self):
        """Scan for SQL injection"""
        url = self.vuln_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Enter URL")
            return
        
        self.vuln_output.insert(tk.END, f"\nScanning {url} for SQL injection...\n")
        self.vuln_output.insert(tk.END, "Testing payloads...\n")
        self.vuln_output.insert(tk.END, "- Testing: ' OR '1'='1\n")
        self.vuln_output.insert(tk.END, "- Testing: ' OR 1=1--\n")
        self.vuln_output.insert(tk.END, "- Testing: admin' --\n")
        self.vuln_output.insert(tk.END, "Scan complete.\n")
    
    def scan_xss(self):
        """Scan for XSS"""
        url = self.vuln_url.get().strip()
        if not url:
            messagebox.showerror("Error", "Enter URL")
            return
        
        self.vuln_output.insert(tk.END, f"\nScanning {url} for XSS...\n")
        self.vuln_output.insert(tk.END, "Testing payloads...\n")
        self.vuln_output.insert(tk.END, "- Testing: <script>alert('XSS')</script>\n")
        self.vuln_output.insert(tk.END, "- Testing: <img src=x onerror=alert('XSS')>\n")
        self.vuln_output.insert(tk.END, "- Testing: <svg onload=alert('XSS')>\n")
        self.vuln_output.insert(tk.END, "Scan complete.\n")
    
    def create_shodan_tab(self):
        """Shodan tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔍 Shodan")
        
        header = tk.Frame(tab, bg=COLORS['accent_red'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🔍 Shodan OSINT",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_red'],
                fg=COLORS['text_white']).pack(pady=15)
        
        if not SHODAN_AVAILABLE:
            tk.Label(tab, text="⚠ Shodan library not installed\nRun: pip install shodan",
                    font=('Arial', 12, 'bold'), fg=COLORS['warning'],
                    bg=COLORS['bg_medium']).pack(pady=50)
            return
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="API Key:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.shodan_key = tk.Entry(input_frame, width=50, show='*')
        self.shodan_key.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(input_frame, text="Query:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.shodan_query = tk.Entry(input_frame, width=50)
        self.shodan_query.grid(row=1, column=1, padx=5, pady=5)
        self.shodan_query.insert(0, "apache")
        
        tk.Button(input_frame, text="🔍 Search", command=self.do_shodan_search,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 11, 'bold')).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.shodan_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                       bg=COLORS['bg_dark'],
                                                       fg='#ff6b6b',
                                                       font=('Courier', 9))
        self.shodan_output.pack(fill='both', expand=True)
    
    def do_shodan_search(self):
        """Shodan search"""
        api_key = self.shodan_key.get().strip()
        query = self.shodan_query.get().strip()
        
        if not api_key or not query:
            messagebox.showerror("Error", "Enter API key and query")
            return
        
        self.shodan_output.insert(tk.END, f"\nSearching Shodan for: {query}...\n")
        threading.Thread(target=self._shodan_thread, args=(api_key, query), daemon=True).start()
    
    def _shodan_thread(self, api_key, query):
        """Shodan thread"""
        result = ShodanSearcher.search(api_key, query)
        if result['success']:
            out = f"\n{'='*70}\nFound {result['total']} results\n{'='*70}\n"
            for device in result['devices']:
                out += f"\nIP: {device['ip']}\n"
                out += f"Port: {device['port']}\n"
                out += f"Org: {device['org']}\n"
                out += f"Location: {device['location']}\n"
                out += f"Banner: {device['banner']}\n"
                out += "-" * 70 + "\n"
            self.master.after(0, lambda: self.shodan_output.insert(tk.END, out))
        else:
            self.master.after(0, lambda: self.shodan_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def create_nmap_tab(self):
        """Nmap tab with comprehensive options"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🛡 Nmap")
        
        header = tk.Frame(tab, bg=COLORS['accent_blue'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🛡 Advanced Network Port Scanner (Nmap)",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_blue'],
                fg=COLORS['text_white']).pack(pady=15)
        
        if not NMAP_AVAILABLE:
            tk.Label(tab, text="⚠ Nmap library not installed\nRun: pip install python-nmap\n\nAlso install Nmap system tool:\n  Ubuntu/Debian: sudo apt-get install nmap\n  CentOS/RHEL: sudo yum install nmap\n  macOS: brew install nmap\n  Windows: Download from nmap.org",
                    font=('Arial', 12, 'bold'), fg=COLORS['warning'],
                    bg=COLORS['bg_medium']).pack(pady=50)
            return
        
        # Input frame
        input_frame = ttk.LabelFrame(tab, text="Scan Configuration", padding=10)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Target
        tk.Label(input_frame, text="Target:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.nmap_target = tk.Entry(input_frame, width=40, font=('Arial', 10))
        self.nmap_target.grid(row=0, column=1, columnspan=3, padx=5, pady=3, sticky='ew')
        self.nmap_target.insert(0, "scanme.nmap.org")
        
        # Port range
        tk.Label(input_frame, text="Ports:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.nmap_ports = tk.Entry(input_frame, width=20, font=('Arial', 10))
        self.nmap_ports.grid(row=1, column=1, padx=5, pady=3)
        self.nmap_ports.insert(0, "1-1000")
        tk.Label(input_frame, text="(e.g., 80, 1-1000, -)", font=('Arial', 8, 'italic'),
                bg=COLORS['bg_medium'], fg=COLORS['text_gray']).grid(row=1, column=2, columnspan=2, sticky='w', padx=5)
        
        # Scan Type
        scan_options_frame = tk.LabelFrame(input_frame, text="Scan Options", 
                                          bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                                          font=('Arial', 9, 'bold'), bd=2)
        scan_options_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=10, sticky='ew')
        
        # Scan Type dropdown
        tk.Label(scan_options_frame, text="Scan Type:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.nmap_scan_type = ttk.Combobox(scan_options_frame, width=25, font=('Arial', 9), state='readonly')
        self.nmap_scan_type.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        self.nmap_scan_type['values'] = [
            '-sS (SYN Stealth Scan)',
            '-sT (TCP Connect Scan)',
            '-sU (UDP Scan)',
            '-sA (ACK Scan)',
            '-sW (Window Scan)',
            '-sM (Maimon Scan)',
            '-sN (Null Scan)',
            '-sF (FIN Scan)',
            '-sX (Xmas Scan)',
            '-sn (Ping Scan - No Port)',
            '-sV (Version Detection)',
            '-sS -sV (Stealth + Version)',
        ]
        self.nmap_scan_type.current(10)  # Default to -sV
        
        # Timing template
        tk.Label(scan_options_frame, text="Timing:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.nmap_timing = ttk.Combobox(scan_options_frame, width=25, font=('Arial', 9), state='readonly')
        self.nmap_timing.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        self.nmap_timing['values'] = [
            '-T0 (Paranoid - Very Slow)',
            '-T1 (Sneaky - Slow)',
            '-T2 (Polite - Slower)',
            '-T3 (Normal - Default)',
            '-T4 (Aggressive - Fast)',
            '-T5 (Insane - Very Fast)',
        ]
        self.nmap_timing.current(3)  # Default to -T3
        
        # Additional options checkboxes
        options_check_frame = tk.Frame(scan_options_frame, bg=COLORS['bg_medium'])
        options_check_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        self.nmap_option_os = tk.BooleanVar(value=False)
        tk.Checkbutton(options_check_frame, text="-O (OS Detection)", variable=self.nmap_option_os,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'], 
                      selectcolor=COLORS['bg_dark'], font=('Arial', 9)).pack(side='left', padx=5)
        
        self.nmap_option_aggressive = tk.BooleanVar(value=False)
        tk.Checkbutton(options_check_frame, text="-A (Aggressive)", variable=self.nmap_option_aggressive,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                      selectcolor=COLORS['bg_dark'], font=('Arial', 9)).pack(side='left', padx=5)
        
        self.nmap_option_verbose = tk.BooleanVar(value=False)
        tk.Checkbutton(options_check_frame, text="-v (Verbose)", variable=self.nmap_option_verbose,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                      selectcolor=COLORS['bg_dark'], font=('Arial', 9)).pack(side='left', padx=5)
        
        self.nmap_option_scripts = tk.BooleanVar(value=False)
        tk.Checkbutton(options_check_frame, text="-sC (Default Scripts)", variable=self.nmap_option_scripts,
                      bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                      selectcolor=COLORS['bg_dark'], font=('Arial', 9)).pack(side='left', padx=5)
        
        # Quick presets
        preset_frame = tk.LabelFrame(input_frame, text="Quick Presets", 
                                     bg=COLORS['bg_medium'], fg=COLORS['text_white'],
                                     font=('Arial', 9, 'bold'), bd=2)
        preset_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky='ew')
        
        tk.Button(preset_frame, text="🚀 Quick Scan", command=lambda: self.nmap_set_preset('quick'),
                 bg=COLORS['info'], fg='white', font=('Arial', 8, 'bold'), width=12).pack(side='left', padx=3, pady=5)
        tk.Button(preset_frame, text="🎯 Intense Scan", command=lambda: self.nmap_set_preset('intense'),
                 bg=COLORS['warning'], fg='white', font=('Arial', 8, 'bold'), width=12).pack(side='left', padx=3, pady=5)
        tk.Button(preset_frame, text="🔍 Comprehensive", command=lambda: self.nmap_set_preset('comprehensive'),
                 bg=COLORS['danger'], fg='white', font=('Arial', 8, 'bold'), width=12).pack(side='left', padx=3, pady=5)
        tk.Button(preset_frame, text="👻 Stealth", command=lambda: self.nmap_set_preset('stealth'),
                 bg=COLORS['accent_purple'], fg='white', font=('Arial', 8, 'bold'), width=12).pack(side='left', padx=3, pady=5)
        
        # Command preview
        tk.Label(input_frame, text="Command:", font=('Arial', 9, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.nmap_command_preview = tk.Entry(input_frame, font=('Courier', 9), 
                                             bg=COLORS['bg_dark'], fg=COLORS['accent_cyan'],
                                             state='readonly')
        self.nmap_command_preview.grid(row=4, column=1, columnspan=3, padx=5, pady=5, sticky='ew')
        
        # Update preview when options change
        self.nmap_scan_type.bind('<<ComboboxSelected>>', lambda e: self.update_nmap_preview())
        self.nmap_timing.bind('<<ComboboxSelected>>', lambda e: self.update_nmap_preview())
        self.nmap_option_os.trace('w', lambda *args: self.update_nmap_preview())
        self.nmap_option_aggressive.trace('w', lambda *args: self.update_nmap_preview())
        self.nmap_option_verbose.trace('w', lambda *args: self.update_nmap_preview())
        self.nmap_option_scripts.trace('w', lambda *args: self.update_nmap_preview())
        
        # Scan button
        tk.Button(input_frame, text="🔍 Start Scan", command=self.do_nmap_scan,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 12, 'bold'),
                 height=2).grid(row=5, column=0, columnspan=4, pady=10, sticky='ew')
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Scan Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.nmap_output = scrolledtext.ScrolledText(output_frame, height=20,
                                                     bg=COLORS['bg_dark'],
                                                     fg='#4ecdc4',
                                                     font=('Courier', 9))
        self.nmap_output.pack(fill='both', expand=True)
        
        # Initial preview
        self.update_nmap_preview()
    
    def nmap_set_preset(self, preset):
        """Set nmap scan preset"""
        if preset == 'quick':
            # Quick scan: SYN scan, fast timing, common ports
            self.nmap_scan_type.set('-sS (SYN Stealth Scan)')
            self.nmap_timing.set('-T4 (Aggressive - Fast)')
            self.nmap_ports.delete(0, tk.END)
            self.nmap_ports.insert(0, '1-1000')
            self.nmap_option_os.set(False)
            self.nmap_option_aggressive.set(False)
            self.nmap_option_verbose.set(False)
            self.nmap_option_scripts.set(False)
        elif preset == 'intense':
            # Intense: SYN + Version, aggressive timing, all ports
            self.nmap_scan_type.set('-sS -sV (Stealth + Version)')
            self.nmap_timing.set('-T4 (Aggressive - Fast)')
            self.nmap_ports.delete(0, tk.END)
            self.nmap_ports.insert(0, '-')
            self.nmap_option_os.set(False)
            self.nmap_option_aggressive.set(False)
            self.nmap_option_verbose.set(True)
            self.nmap_option_scripts.set(True)
        elif preset == 'comprehensive':
            # Comprehensive: Aggressive scan, all features
            self.nmap_scan_type.set('-sV (Version Detection)')
            self.nmap_timing.set('-T4 (Aggressive - Fast)')
            self.nmap_ports.delete(0, tk.END)
            self.nmap_ports.insert(0, '-')
            self.nmap_option_os.set(True)
            self.nmap_option_aggressive.set(True)
            self.nmap_option_verbose.set(True)
            self.nmap_option_scripts.set(True)
        elif preset == 'stealth':
            # Stealth: Slow, sneaky scan
            self.nmap_scan_type.set('-sS (SYN Stealth Scan)')
            self.nmap_timing.set('-T1 (Sneaky - Slow)')
            self.nmap_ports.delete(0, tk.END)
            self.nmap_ports.insert(0, '1-1000')
            self.nmap_option_os.set(False)
            self.nmap_option_aggressive.set(False)
            self.nmap_option_verbose.set(False)
            self.nmap_option_scripts.set(False)
        
        self.update_nmap_preview()
    
    def update_nmap_preview(self):
        """Update nmap command preview"""
        args = []
        
        # Scan type
        scan_type = self.nmap_scan_type.get().split(' ')[0]
        args.append(scan_type)
        
        # Timing
        timing = self.nmap_timing.get().split(' ')[0]
        args.append(timing)
        
        # Additional options
        if self.nmap_option_os.get():
            args.append('-O')
        if self.nmap_option_aggressive.get():
            args.append('-A')
        if self.nmap_option_verbose.get():
            args.append('-v')
        if self.nmap_option_scripts.get():
            args.append('-sC')
        
        # Ports
        ports = self.nmap_ports.get().strip()
        if ports and ports != '-':
            args.append(f'-p {ports}')
        
        # Target
        target = self.nmap_target.get().strip()
        
        cmd = f"nmap {' '.join(args)} {target}"
        
        self.nmap_command_preview.config(state='normal')
        self.nmap_command_preview.delete(0, tk.END)
        self.nmap_command_preview.insert(0, cmd)
        self.nmap_command_preview.config(state='readonly')
    
    def do_nmap_scan(self):
        """Nmap scan with selected options"""
        target = self.nmap_target.get().strip()
        if not target:
            messagebox.showerror("Error", "Enter target")
            return
        
        # Build arguments
        args = []
        
        # Scan type
        scan_type = self.nmap_scan_type.get().split(' ')[0]
        args.append(scan_type)
        
        # Timing
        timing = self.nmap_timing.get().split(' ')[0]
        args.append(timing)
        
        # Ports
        ports = self.nmap_ports.get().strip()
        port_arg = None
        if ports and ports != '-':
            port_arg = ports
        
        # Additional options
        if self.nmap_option_os.get():
            args.append('-O')
        if self.nmap_option_aggressive.get():
            args.append('-A')
        if self.nmap_option_verbose.get():
            args.append('-v')
        if self.nmap_option_scripts.get():
            args.append('-sC')
        
        arguments = ' '.join(args)
        
        self.nmap_output.insert(tk.END, f"\n{'='*70}\n")
        self.nmap_output.insert(tk.END, f"Scanning {target} with: nmap {arguments}")
        if port_arg:
            self.nmap_output.insert(tk.END, f" -p {port_arg}")
        self.nmap_output.insert(tk.END, f"\n{'='*70}\n")
        
        threading.Thread(target=self._nmap_thread, args=(target, arguments, port_arg), daemon=True).start()
    
    def _nmap_thread(self, target, arguments, ports=None):
        """Nmap thread with custom arguments"""
        result = NmapScanner.scan(target, arguments, ports)
        if result['success']:
            out = f"\n{'='*70}\nScan Results\n{'='*70}\n"
            for host, data in result['results'].items():
                out += f"\nHost: {host} ({data['state']})\n"
                
                # OS detection if available
                if 'osmatch' in data:
                    out += f"\nOS Detection:\n"
                    for os in data['osmatch'][:3]:  # Top 3 matches
                        out += f"  • {os['name']} (Accuracy: {os['accuracy']}%)\n"
                
                # Ports
                out += f"\nOpen Ports:\n"
                for proto, ports in data['protocols'].items():
                    for port, info in ports.items():
                        out += f"  {port}/{proto} - {info['state']} - {info['name']}"
                        if 'version' in info and info['version']:
                            out += f" ({info['version']})"
                        out += "\n"
                out += "-" * 70 + "\n"
            self.master.after(0, lambda: self.nmap_output.insert(tk.END, out))
            self.nmap_last_results = out
        else:
            error_msg = f"\n❌ Error: {result['error']}\n"
            if 'permission' in result['error'].lower() or 'must be root' in result['error'].lower():
                error_msg += "\n⚠️  Some scan types (like -sS SYN scan) require root/admin privileges.\n"
                error_msg += "Try: sudo python3 cybersecurity_toolkit_ML_ENHANCED.py\n"
                error_msg += "Or use -sT (TCP Connect Scan) which doesn't require root.\n"
            self.master.after(0, lambda: self.nmap_output.insert(tk.END, error_msg))
    
    def create_whois_tab(self):
        """WHOIS tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="📋 WHOIS")
        
        header = tk.Frame(tab, bg=COLORS['accent_purple'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="📋 WHOIS Lookup",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_purple'],
                fg=COLORS['text_white']).pack(pady=15)
        
        if not WHOIS_AVAILABLE:
            tk.Label(tab, text="⚠ WHOIS library not installed\nRun: pip install python-whois",
                    font=('Arial', 12, 'bold'), fg=COLORS['warning'],
                    bg=COLORS['bg_medium']).pack(pady=50)
            return
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Domain:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.whois_domain = tk.Entry(input_frame, width=40, font=('Arial', 10))
        self.whois_domain.grid(row=0, column=1, padx=5)
        self.whois_domain.insert(0, "google.com")
        
        tk.Button(input_frame, text="🔍 Lookup", command=self.do_whois,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.whois_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                      bg=COLORS['bg_dark'],
                                                      fg='#a29bfe',
                                                      font=('Courier', 9))
        self.whois_output.pack(fill='both', expand=True)
    
    def do_whois(self):
        """WHOIS lookup"""
        domain = self.whois_domain.get().strip()
        if not domain:
            messagebox.showerror("Error", "Enter domain")
            return
        
        self.whois_output.insert(tk.END, f"\nLooking up {domain}...\n")
        threading.Thread(target=self._whois_thread, args=(domain,), daemon=True).start()
    
    def _whois_thread(self, domain):
        """WHOIS thread"""
        result = WhoisLookup.lookup(domain)
        if result['success']:
            out = f"\n{'='*70}\nWHOIS: {domain}\n{'='*70}\n"
            out += f"Domain: {result['domain']}\n"
            out += f"Registrar: {result['registrar']}\n"
            out += f"Created: {result['creation_date']}\n"
            out += f"Expires: {result['expiration_date']}\n"
            out += f"Name Servers: {result['name_servers']}\n"
            out += "-" * 70 + "\n"
            self.master.after(0, lambda: self.whois_output.insert(tk.END, out))
            self.whois_last_results = out
        else:
            self.master.after(0, lambda: self.whois_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def create_dns_tab(self):
        """DNS tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🌍 DNS")
        
        header = tk.Frame(tab, bg=COLORS['accent_cyan'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🌍 DNS Analyzer",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_cyan'],
                fg=COLORS['text_white']).pack(pady=15)
        
        if not DNS_AVAILABLE:
            tk.Label(tab, text="⚠ DNS library not installed\nRun: pip install dnspython",
                    font=('Arial', 12, 'bold'), fg=COLORS['warning'],
                    bg=COLORS['bg_medium']).pack(pady=50)
            return
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Domain:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.dns_domain = tk.Entry(input_frame, width=40, font=('Arial', 10))
        self.dns_domain.grid(row=0, column=1, padx=5)
        self.dns_domain.insert(0, "google.com")
        
        tk.Button(input_frame, text="🔍 Analyze", command=self.do_dns,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=2, pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.dns_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                    bg=COLORS['bg_dark'],
                                                    fg='#00d2d3',
                                                    font=('Courier', 9))
        self.dns_output.pack(fill='both', expand=True)
    
    def do_dns(self):
        """DNS analysis"""
        domain = self.dns_domain.get().strip()
        if not domain:
            messagebox.showerror("Error", "Enter domain")
            return
        
        self.dns_output.insert(tk.END, f"\nAnalyzing DNS for {domain}...\n")
        threading.Thread(target=self._dns_thread, args=(domain,), daemon=True).start()
    
    def _dns_thread(self, domain):
        """DNS thread"""
        result = DNSAnalyzer.analyze(domain)
        if result['success']:
            out = f"\n{'='*70}\nDNS: {domain}\n{'='*70}\n"
            for rtype, records in result['results'].items():
                out += f"\n{rtype} Records:\n"
                if records:
                    for record in records:
                        out += f"  {record}\n"
                else:
                    out += "  (none)\n"
            self.master.after(0, lambda: self.dns_output.insert(tk.END, out))
            self.dns_last_results = out
        else:
            self.master.after(0, lambda: self.dns_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def create_ssl_tab(self):
        """SSL tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🔒 SSL/TLS")
        
        header = tk.Frame(tab, bg=COLORS['accent_green'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🔒 SSL/TLS Analyzer",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_green'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Configuration", padding=15)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="Hostname:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.ssl_hostname = tk.Entry(input_frame, width=35, font=('Arial', 10))
        self.ssl_hostname.grid(row=0, column=1, padx=5)
        self.ssl_hostname.insert(0, "www.google.com")
        
        tk.Label(input_frame, text="Port:", font=('Arial', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=2, sticky='w', padx=5)
        self.ssl_port = tk.Entry(input_frame, width=10)
        self.ssl_port.grid(row=0, column=3, padx=5)
        self.ssl_port.insert(0, "443")
        
        tk.Button(input_frame, text="🔍 Analyze", command=self.do_ssl,
                 bg=COLORS['accent_green'], fg='white', font=('Arial', 11, 'bold')).grid(row=1, column=0, columnspan=4, pady=10)
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="Results", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ssl_output = scrolledtext.ScrolledText(output_frame, height=25,
                                                    bg=COLORS['bg_dark'],
                                                    fg='#2ecc71',
                                                    font=('Courier', 9))
        self.ssl_output.pack(fill='both', expand=True)
    
    def do_ssl(self):
        """SSL analysis"""
        hostname = self.ssl_hostname.get().strip()
        port = int(self.ssl_port.get())
        
        if not hostname:
            messagebox.showerror("Error", "Enter hostname")
            return
        
        self.ssl_output.insert(tk.END, f"\nAnalyzing {hostname}:{port}...\n")
        threading.Thread(target=self._ssl_thread, args=(hostname, port), daemon=True).start()
    
    def _ssl_thread(self, hostname, port):
        """SSL thread"""
        result = SSLAnalyzer.analyze(hostname, port)
        if result['success']:
            out = f"\n{'='*70}\nSSL/TLS: {hostname}\n{'='*70}\n"
            out += f"Subject: {result['subject']}\n"
            out += f"Issuer: {result['issuer']}\n"
            out += f"Valid From: {result['not_before']}\n"
            out += f"Valid Until: {result['not_after']}\n"
            out += f"Version: {result['version']}\n"
            out += f"Serial: {result['serial']}\n"
            out += f"Cipher: {result['cipher']}\n"
            out += f"TLS Version: {result['tls_version']}\n"
            self.master.after(0, lambda: self.ssl_output.insert(tk.END, out))
            self.ssl_last_results = out
        else:
            self.master.after(0, lambda: self.ssl_output.insert(tk.END, f"Error: {result['error']}\n"))
    
    def create_ai_tab(self):
        """AI analysis tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="🤖 AI Analysis")
        
        header = tk.Frame(tab, bg=COLORS['accent_purple'], height=60)
        header.pack(fill='x')
        tk.Label(header, text="🤖 AI-Powered Security Analysis",
                font=('Arial', 16, 'bold'), bg=COLORS['accent_purple'],
                fg=COLORS['text_white']).pack(pady=15)
        
        # API config
        api_frame = ttk.LabelFrame(tab, text="API Configuration", padding=15)
        api_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(api_frame, text="Provider:", font=('Arial', 10, 'bold'),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=0, column=0, sticky='w', padx=5)
        self.ai_provider = tk.StringVar(value="chatgpt")
        ttk.Radiobutton(api_frame, text="ChatGPT", variable=self.ai_provider, value="chatgpt").grid(row=0, column=1, padx=5)
        ttk.Radiobutton(api_frame, text="Claude", variable=self.ai_provider, value="claude").grid(row=0, column=2, padx=5)
        
        tk.Label(api_frame, text="ChatGPT Key:", font=('Arial', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.chatgpt_key = tk.Entry(api_frame, width=50, show='*', font=('Arial', 9))
        self.chatgpt_key.grid(row=1, column=1, columnspan=2, sticky='ew', padx=5)
        
        tk.Label(api_frame, text="Claude Key:", font=('Arial', 10),
                bg=COLORS['bg_medium'], fg=COLORS['text_white']).grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.claude_key = tk.Entry(api_frame, width=50, show='*', font=('Arial', 9))
        self.claude_key.grid(row=2, column=1, columnspan=2, sticky='ew', padx=5)
        
        # Input
        input_frame = ttk.LabelFrame(tab, text="Scan Results", padding=10)
        input_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.ai_input = scrolledtext.ScrolledText(input_frame, height=10, font=('Courier', 9))
        self.ai_input.pack(fill='both', expand=True, padx=5, pady=5)
        
        tk.Button(input_frame, text="🤖 Analyze with AI", command=self.do_ai_analyze,
                 bg=COLORS['accent_purple'], fg='white', font=('Arial', 11, 'bold'),
                 height=2).pack(pady=5, fill='x')
        
        # Output
        output_frame = ttk.LabelFrame(tab, text="AI Analysis", padding=5)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ai_output = scrolledtext.ScrolledText(output_frame, height=15,
                                                   bg=COLORS['bg_dark'],
                                                   fg=COLORS['accent_purple'],
                                                   font=('Courier', 9))
        self.ai_output.pack(fill='both', expand=True)
    
    def do_ai_analyze(self):
        """AI analysis"""
        scan_results = self.ai_input.get(1.0, tk.END).strip()
        if not scan_results:
            messagebox.showerror("Error", "Enter scan results")
            return
        
        provider = self.ai_provider.get()
        if provider == "chatgpt":
            api_key = self.chatgpt_key.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Enter ChatGPT API key")
                return
            self.ai_output.insert(tk.END, "\nAnalyzing with ChatGPT...\n")
            threading.Thread(target=self._ai_chatgpt_thread, args=(api_key, scan_results), daemon=True).start()
        else:
            api_key = self.claude_key.get().strip()
            if not api_key:
                messagebox.showerror("Error", "Enter Claude API key")
                return
            self.ai_output.insert(tk.END, "\nAnalyzing with Claude...\n")
            threading.Thread(target=self._ai_claude_thread, args=(api_key, scan_results), daemon=True).start()
    
    def _ai_chatgpt_thread(self, api_key, results):
        """ChatGPT thread"""
        result = AIAnalyzer.analyze_with_chatgpt(api_key, results)
        if result['success']:
            self.master.after(0, lambda: self.ai_output.insert(tk.END, f"\n{'='*70}\n{result['analysis']}\n"))
        else:
            self.master.after(0, lambda: self.ai_output.insert(tk.END, f"\nError: {result['error']}\n"))
    
    def _ai_claude_thread(self, api_key, results):
        """Claude thread"""
        result = AIAnalyzer.analyze_with_claude(api_key, results)
        if result['success']:
            self.master.after(0, lambda: self.ai_output.insert(tk.END, f"\n{'='*70}\n{result['analysis']}\n"))
        else:
            self.master.after(0, lambda: self.ai_output.insert(tk.END, f"\nError: {result['error']}\n"))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║   Integrated Cybersecurity Toolkit v6.0 - ML Enhanced        ║")
    print("║   Advanced Password Analysis + Network Security Tools         ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    print("✓ ML-Based Password Strength Analysis")
    print("✓ Password Hash Cracking (MD5, SHA, NTLM)")
    print("✓ Network Tools (Ping, Traceroute, Port Scan)")
    print("✓ Shodan OSINT Integration")
    print("✓ Nmap Network Scanner")
    print("✓ WHOIS Lookup")
    print("✓ DNS Analysis")
    print("✓ SSL/TLS Certificate Analyzer")
    print("✓ AI-Powered Vulnerability Analysis")
    print("\n⚠️  EDUCATIONAL USE ONLY - Unauthorized use is illegal!\n")
    
    if not ML_AVAILABLE:
        print("⚠️  Note: ML libraries not installed (pandas, joblib)")
        print("   For full ML features, run: pip install pandas joblib scikit-learn\n")
    
    if not SHODAN_AVAILABLE:
        print("⚠️  Note: Shodan library not installed")
        print("   To enable Shodan: pip install shodan\n")
    
    root = tk.Tk()
    app = CyberSecurityToolkit_GUI(root)
    root.mainloop()
