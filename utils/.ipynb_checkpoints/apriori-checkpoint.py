"""Apriori algorithm for rare pattern mining."""

import numpy as np
from collections import defaultdict
from itertools import combinations
from typing import List, Set, Dict, Tuple, FrozenSet
from tqdm import tqdm


class AprioriMiner:
    """
    Apriori algorithm for mining frequent and rare itemsets
    
    Optimized for finding RARE patterns (low support) for anomaly detection
    """
    
    def __init__(self, min_support=0.001, max_support=0.05, verbose=True):
        """
        Args:
            min_support: Minimum support threshold (e.g., 0.001 = 0.1%)
            max_support: Maximum support for "rare" patterns (e.g., 0.05 = 5%)
            verbose: Print progress
        """
        self.min_support = min_support
        self.max_support = max_support
        self.verbose = verbose
        
    def mine_itemsets(self, transactions: List[Set[str]], max_length=3):
        """
        Mine frequent itemsets using Apriori algorithm
        
        Args:
            transactions: List of sets, each set is a transaction (items in that transaction)
            max_length: Maximum itemset size to mine
            
        Returns:
            all_itemsets: Dict[frozenset, float] - itemset -> support
        """
        n_transactions = len(transactions)
        
        if self.verbose:
            print(f"Mining itemsets with Apriori algorithm")
            print(f"   Transactions: {n_transactions:,}")
            print(f"   Support range: [{self.min_support}, {self.max_support}]")
            print(f"   Max itemset size: {max_length}")
        
        # Step 1: Find frequent 1-itemsets
        if self.verbose:
            print(f"   Counting 1-itemsets...")
        
        item_counts = defaultdict(int)
        for transaction in tqdm(transactions, desc="Scanning transactions", disable=not self.verbose):
            for item in transaction:
                item_counts[item] += 1
        
        # Convert to support and filter
        all_itemsets = {}
        frequent_itemsets = {}
        
        for item, count in item_counts.items():
            support = count / n_transactions
            itemset = frozenset([item])
            all_itemsets[itemset] = support
            
            # Keep itemsets above min_support for generating candidates
            if support >= self.min_support:
                frequent_itemsets[itemset] = support
        
        if self.verbose:
            print(f"   1-itemsets: {len(frequent_itemsets):,} frequent (support >= {self.min_support})")
        
        # Step 2: Iteratively find larger itemsets
        current_itemsets = frequent_itemsets
        k = 2
        
        while current_itemsets and k <= max_length:
            # Generate candidates
            candidates = self._generate_candidates(current_itemsets, k)
            
            if not candidates:
                break
            
            # Count support for candidates
            candidate_counts = defaultdict(int)
            for transaction in tqdm(transactions, desc=f"Checking {k}-itemsets", disable=not self.verbose):
                for candidate in candidates:
                    if candidate.issubset(transaction):
                        candidate_counts[candidate] += 1
            
            # Filter by support
            current_itemsets = {}
            for candidate, count in candidate_counts.items():
                support = count / n_transactions
                all_itemsets[candidate] = support
                
                if support >= self.min_support:
                    current_itemsets[candidate] = support
            
            if self.verbose:
                print(f"   {k}-itemsets: {len(current_itemsets):,} frequent")
            
            k += 1
        
        if self.verbose:
            print(f"Total itemsets found: {len(all_itemsets):,}")
        
        return all_itemsets
    
    def filter_rare_itemsets(self, all_itemsets: Dict[FrozenSet, float]):
        """Filter itemsets to keep only rare ones."""
        rare_itemsets = [
            (itemset, support) 
            for itemset, support in all_itemsets.items()
            if self.min_support <= support <= self.max_support
        ]
        
        # Sort by support (rarest first)
        rare_itemsets.sort(key=lambda x: x[1])
        
        if self.verbose:
            print(f"\nRare itemsets (support in [{self.min_support}, {self.max_support}]):")
            print(f"   Count: {len(rare_itemsets):,}")
            if rare_itemsets:
                print(f"   Rarest: {rare_itemsets[0][1]:.4f} support")
                print(f"   Least rare: {rare_itemsets[-1][1]:.4f} support")
        
        return rare_itemsets
    
    def mine_rare_itemsets(self, transactions: List[Set[str]], max_length=3):
        """
        Convenience method: Mine and return only rare itemsets
        
        Args:
            transactions: List of sets
            max_length: Maximum itemset size
            
        Returns:
            rare_itemsets: List of (itemset, support) tuples
        """
        all_itemsets = self.mine_itemsets(transactions, max_length)
        rare_itemsets = self.filter_rare_itemsets(all_itemsets)
        return rare_itemsets
    
    def _generate_candidates(self, frequent_itemsets: Dict[FrozenSet, float], k: int):
        """Generate k-itemset candidates from (k-1)-itemsets."""
        items = set()
        for itemset in frequent_itemsets.keys():
            items.update(itemset)
        
        # Generate all k-combinations
        candidates = set()
        items_list = sorted(list(items))
        
        for combo in combinations(items_list, k):
            candidate = frozenset(combo)
            
            # Prune: Check if all (k-1)-subsets are frequent
            subsets = [frozenset(s) for s in combinations(combo, k-1)]
            if all(subset in frequent_itemsets for subset in subsets):
                candidates.add(candidate)
        
        return candidates


class AssociationRuleMiner:
    """Generate association rules from itemsets."""
    
    def __init__(self, min_confidence=0.5, verbose=True):
        """
        Args:
            min_confidence: Minimum confidence for rules
            verbose: Print progress
        """
        self.min_confidence = min_confidence
        self.verbose = verbose
    
    def generate_rules(self, itemsets: List[Tuple[FrozenSet, float]], 
                      transactions: List[Set[str]]):
        """
        Generate association rules from itemsets
        
        Args:
            itemsets: List of (itemset, support) tuples
            transactions: Original transactions (for confidence calculation)
            
        Returns:
            rules: List of (antecedent, consequent, confidence, support) tuples
        """
        n_transactions = len(transactions)
        rules = []
        
        if self.verbose:
            print(f"\nGenerating association rules")
            print(f"   Min confidence: {self.min_confidence}")
        
        # For each itemset with size >= 2
        for itemset, support in itemsets:
            if len(itemset) < 2:
                continue
            
            # Try all possible non-empty subsets as antecedents
            items = list(itemset)
            for r in range(1, len(items)):
                for antecedent_items in combinations(items, r):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset - antecedent
                    
                    if not consequent:  # Skip if consequent is empty
                        continue
                    
                    # Calculate confidence: support(itemset) / support(antecedent)
                    # Count antecedent support
                    antecedent_count = sum(
                        1 for t in transactions if antecedent.issubset(t)
                    )
                    
                    if antecedent_count > 0:
                        confidence = (support * n_transactions) / antecedent_count
                        
                        if confidence >= self.min_confidence:
                            rules.append((antecedent, consequent, confidence, support))
        
        if self.verbose:
            print(f"Generated {len(rules):,} rules")
        
        return rules


def mine_rare_patterns(transactions, min_support=0.001, max_support=0.05, 
                      max_length=3, min_confidence=0.5, verbose=True):
    """
    One-stop function to mine rare patterns and generate rules
    
    Args:
        transactions: List of sets
        min_support: Minimum support
        max_support: Maximum support for "rare"
        max_length: Maximum itemset size
        min_confidence: Minimum confidence for rules
        verbose: Print progress
        
    Returns:
        rare_itemsets: List of (itemset, support) tuples
        rules: List of (antecedent, consequent, confidence, support) tuples
    """
    # Mine rare itemsets
    miner = AprioriMiner(min_support, max_support, verbose)
    rare_itemsets = miner.mine_rare_itemsets(transactions, max_length)
    
    # Generate association rules
    rule_miner = AssociationRuleMiner(min_confidence, verbose)
    rules = rule_miner.generate_rules(rare_itemsets, transactions)
    
    return rare_itemsets, rules


def apriori(transactions, min_support=0.001, max_support=0.05, max_length=3, verbose=True):
    """Mine rare itemsets using Apriori algorithm."""
    miner = AprioriMiner(min_support, max_support, verbose)
    return miner.mine_rare_itemsets(transactions, max_length)


def generate_association_rules(rare_itemsets, transactions, min_confidence=0.5, verbose=True):
    """Generate association rules from rare itemsets."""
    rule_miner = AssociationRuleMiner(min_confidence, verbose)
    return rule_miner.generate_rules(rare_itemsets, transactions)
