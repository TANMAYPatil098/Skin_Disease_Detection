import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import sys

class SkinProductRecommender:
    def __init__(self):
        # Updated product database to match specific diseases in predict.py
        self.products_df = pd.DataFrame({
            'product_id': range(1, 11),
            'name': [
                'Gentle Cleanser', 'Hydrating Moisturizer', 'Cortisone Cream',
                'Anti-fungal Cream', 'Salicylic Acid Treatment', 'Aloe Vera Gel',
                'Zinc Oxide Cream', 'Tea Tree Oil', 'Hyaluronic Acid Serum',
                'Niacinamide Solution'
            ],
            'category': [
                'cleanser', 'moisturizer', 'medication',
                'medication', 'treatment', 'treatment',
                'treatment', 'treatment', 'serum', 'serum'
            ],
            'ingredients': [
                ['glycerin', 'aloe', 'chamomile'],
                ['hyaluronic acid', 'ceramides', 'glycerin'],
                ['hydrocortisone', 'aloe'],
                ['miconazole', 'zinc oxide'],
                ['salicylic acid', 'tea tree'],
                ['aloe vera', 'vitamin e'],
                ['zinc oxide', 'titanium dioxide'],
                ['tea tree oil', 'witch hazel'],
                ['hyaluronic acid', 'vitamin b5'],
                ['niacinamide', 'zinc']
            ],
            # Ensure conditions are mapped to diseases in predict.py
            'conditions': [
                ['Chronic_Dermatitis'],     # Gentle Cleanser
                ['Chronic_Dermatitis'],     # Hydrating Moisturizer
                ['Chronic_Dermatitis'],     # Cortisone Cream
                ['Seborrheic_Dermatitis'],  # Anti-fungal Cream
                ['Psoriasis'],              # Salicylic Acid Treatment
                ['Chronic_Dermatitis'],     # Aloe Vera Gel
                ['Psoriasis'],              # Zinc Oxide Cream
                ['Lichen_Planus'],          # Tea Tree Oil
                ['Pityriasis_Rosea'],       # Hyaluronic Acid Serum
                ['Seborrheic_Dermatitis']   # Niacinamide Solution
            ],
            'price': [15, 25, 12, 14, 18, 10, 16, 20, 30, 22]
        })
        
        # Initialize the MultiLabelBinarizer for ingredients and conditions
        self.mlb_ingredients = MultiLabelBinarizer()
        self.mlb_conditions = MultiLabelBinarizer()
        
        # Create feature matrices
        self._prepare_features()

    def _prepare_features(self):
        # Convert ingredients and conditions lists to binary matrices
        ingredients_matrix = self.mlb_ingredients.fit_transform(self.products_df['ingredients'])
        conditions_matrix = self.mlb_conditions.fit_transform(self.products_df['conditions'])
        
        # Convert category to dummy variables
        category_dummies = pd.get_dummies(self.products_df['category'])
        
        # Scale prices
        scaler = StandardScaler()
        prices_scaled = scaler.fit_transform(self.products_df[['price']])
        
        # Combine all features
        self.features_matrix = pd.concat([
            pd.DataFrame(ingredients_matrix, columns=self.mlb_ingredients.classes_),
            pd.DataFrame(conditions_matrix, columns=self.mlb_conditions.classes_),
            category_dummies,
            pd.DataFrame(prices_scaled, columns=['price'])
        ], axis=1)

    def get_recommendations(self, skin_condition, num_recommendations=3, max_price=None):
        """
        Get product recommendations for a specific skin condition.
        
        Parameters:
        skin_condition (str): The predicted skin condition
        num_recommendations (int): Number of products to recommend
        max_price (float): Maximum price filter (optional)
        
        Returns:
        DataFrame: Recommended products with their details
        """
        # Filter products that are suitable for the condition
        suitable_products = self.products_df[
            self.products_df['conditions'].apply(lambda x: skin_condition.lower() in [c.lower() for c in x])
        ]
        
        if max_price is not None:
            suitable_products = suitable_products[suitable_products['price'] <= max_price]
        
        if suitable_products.empty:
            return pd.DataFrame(), f"No products found for the specified condition: {skin_condition} and criteria."
        
        # Get indices of suitable products
        suitable_indices = suitable_products.index
        
        # Calculate similarity scores for suitable products
        similarity_matrix = cosine_similarity(
            self.features_matrix.iloc[suitable_indices],
            self.features_matrix.iloc[suitable_indices]
        )
        
        # Get average similarity scores for each product
        avg_similarity = similarity_matrix.mean(axis=1)
        
        # Sort products by similarity score
        recommended_indices = suitable_indices[avg_similarity.argsort()[::-1][:num_recommendations]]
        
        recommendations = self.products_df.iloc[recommended_indices].copy()
        recommendations['relevance_score'] = avg_similarity[avg_similarity.argsort()[::-1][:num_recommendations]]
        
        return recommendations[['name', 'category', 'price', 'relevance_score']], "Successfully found recommendations."

def main():
    if len(sys.argv) < 2:
        print("Usage: recommender.py <predicted_skin_condition>")
        sys.exit(1)
    
    skin_condition = sys.argv[1]  # Get the skin condition predicted by the model (from predict.py)
    
    recommender = SkinProductRecommender()
    
    # You can optionally specify a maximum price and number of recommendations
    max_price = None
    num_recommendations = 3
    
    recommendations, message = recommender.get_recommendations(skin_condition, num_recommendations, max_price)
    
    if recommendations.empty:
        print(message)
    else:
        print("Recommended products:")
        print(recommendations.to_string(index=False))

if __name__ == "__main__":
    main()
