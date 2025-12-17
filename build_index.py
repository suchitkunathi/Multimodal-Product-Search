
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clip_encoder import CLIPEncoder
from faiss_index import FAISSIndex

# EXPANDED PRODUCT DATABASE - 200+ PRODUCTS
PRODUCTS_DATABASE = [
    # CLOTHING - T-SHIRTS & TOPS
    {"id": "1", "name": "Blue Cotton T-Shirt", "category": "Clothing", "price": 19.99, "color": "blue", "material": "cotton", "description": "Classic blue cotton t-shirt perfect for everyday wear", "image_url": "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400"},
    {"id": "2", "name": "White Linen T-Shirt", "category": "Clothing", "price": 24.99, "color": "white", "material": "linen", "description": "Breathable white linen t-shirt for summer", "image_url": "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400"},
    {"id": "3", "name": "Black Cotton T-Shirt", "category": "Clothing", "price": 19.99, "color": "black", "material": "cotton", "description": "Essential black cotton t-shirt for any wardrobe", "image_url": "https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400"},
    {"id": "4", "name": "Red Striped T-Shirt", "category": "Clothing", "price": 22.99, "color": "red", "material": "cotton blend", "description": "Stylish red and white striped cotton t-shirt", "image_url": "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400"},
    {"id": "5", "name": "Green Polo Shirt", "category": "Clothing", "price": 34.99, "color": "green", "material": "cotton", "description": "Classic green polo shirt with buttons", "image_url": "https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?w=400"},
    {"id": "6", "name": "Navy Blue Polo Shirt", "category": "Clothing", "price": 34.99, "color": "navy", "material": "cotton", "description": "Professional navy blue polo shirt", "image_url": "https://images.unsplash.com/photo-1586790170083-2f9ceadc732d?w=400"},
    {"id": "7", "name": "Vintage Graphic Tee", "category": "Clothing", "price": 29.99, "color": "gray", "material": "cotton", "description": "Retro vintage graphic t-shirt with distressed print", "image_url": "https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400"},
    {"id": "8", "name": "Oversized White Tee", "category": "Clothing", "price": 26.99, "color": "white", "material": "cotton", "description": "Trendy oversized white t-shirt for casual wear", "image_url": "https://images.unsplash.com/photo-1618354691373-d851c5c3a990?w=400"},
    {"id": "9", "name": "Long Sleeve Henley", "category": "Clothing", "price": 39.99, "color": "burgundy", "material": "cotton blend", "description": "Classic burgundy henley with button placket", "image_url": "https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400"},
    {"id": "10", "name": "Muscle Tank Top", "category": "Clothing", "price": 18.99, "color": "black", "material": "cotton", "description": "Athletic black muscle tank for workouts", "image_url": "https://images.unsplash.com/photo-1503341504253-dff4815485f1?w=400"},
    
    # CLOTHING - JEANS & PANTS
    {"id": "11", "name": "Denim Jeans Slim Fit", "category": "Clothing", "price": 59.99, "color": "blue", "material": "denim", "description": "Slim fit blue denim jeans with classic styling", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "12", "name": "Black Denim Jeans", "category": "Clothing", "price": 59.99, "color": "black", "material": "denim", "description": "Black denim jeans for versatile styling", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "13", "name": "Ripped Skinny Jeans", "category": "Clothing", "price": 69.99, "color": "blue", "material": "denim", "description": "Trendy ripped skinny jeans with distressed details", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "14", "name": "High Waist Mom Jeans", "category": "Clothing", "price": 64.99, "color": "light blue", "material": "denim", "description": "Vintage-inspired high waist mom jeans", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "15", "name": "Cargo Pants Khaki", "category": "Clothing", "price": 49.99, "color": "khaki", "material": "cotton twill", "description": "Utility cargo pants with multiple pockets", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "16", "name": "Chino Pants Navy", "category": "Clothing", "price": 44.99, "color": "navy", "material": "cotton", "description": "Classic navy chino pants for smart casual", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "17", "name": "Jogger Sweatpants", "category": "Clothing", "price": 39.99, "color": "gray", "material": "fleece", "description": "Comfortable gray jogger sweatpants", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    {"id": "18", "name": "Dress Pants Black", "category": "Clothing", "price": 79.99, "color": "black", "material": "wool blend", "description": "Formal black dress pants for business", "image_url": "https://images.unsplash.com/photo-1542272604-787c3835535d?w=400"},
    
    # CLOTHING - HOODIES & SWEATSHIRTS
    {"id": "19", "name": "Gray Sweatshirt", "category": "Clothing", "price": 44.99, "color": "gray", "material": "fleece", "description": "Comfortable gray fleece sweatshirt", "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400"},
    {"id": "20", "name": "Black Hoodie", "category": "Clothing", "price": 54.99, "color": "black", "material": "fleece", "description": "Warm black hoodie with kangaroo pocket", "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400"},
    {"id": "21", "name": "Zip-Up Hoodie Navy", "category": "Clothing", "price": 59.99, "color": "navy", "material": "cotton blend", "description": "Navy zip-up hoodie with drawstring", "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400"},
    {"id": "22", "name": "Oversized Hoodie Pink", "category": "Clothing", "price": 49.99, "color": "pink", "material": "fleece", "description": "Trendy oversized pink hoodie", "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400"},
    {"id": "23", "name": "Pullover Sweatshirt", "category": "Clothing", "price": 42.99, "color": "burgundy", "material": "cotton", "description": "Classic burgundy pullover sweatshirt", "image_url": "https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400"},
    
    # CLOTHING - SHIRTS & BLOUSES
    {"id": "24", "name": "White Button Down Shirt", "category": "Clothing", "price": 49.99, "color": "white", "material": "cotton", "description": "Classic white button-down dress shirt", "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
    {"id": "25", "name": "Flannel Shirt Plaid", "category": "Clothing", "price": 39.99, "color": "red", "material": "cotton", "description": "Cozy red plaid flannel shirt", "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
    {"id": "26", "name": "Denim Shirt Light Blue", "category": "Clothing", "price": 44.99, "color": "light blue", "material": "denim", "description": "Casual light blue denim shirt", "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
    {"id": "27", "name": "Silk Blouse Black", "category": "Clothing", "price": 79.99, "color": "black", "material": "silk", "description": "Elegant black silk blouse", "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
    {"id": "28", "name": "Hawaiian Shirt Tropical", "category": "Clothing", "price": 34.99, "color": "multicolor", "material": "rayon", "description": "Vibrant tropical print Hawaiian shirt", "image_url": "https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400"},
    
    # CLOTHING - DRESSES & SKIRTS
    {"id": "29", "name": "Little Black Dress", "category": "Clothing", "price": 89.99, "color": "black", "material": "polyester", "description": "Classic little black dress for any occasion", "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400"},
    {"id": "30", "name": "Floral Summer Dress", "category": "Clothing", "price": 69.99, "color": "floral", "material": "cotton", "description": "Light floral print summer dress", "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400"},
    {"id": "31", "name": "Maxi Dress Bohemian", "category": "Clothing", "price": 79.99, "color": "earth tones", "material": "rayon", "description": "Flowing bohemian maxi dress", "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400"},
    {"id": "32", "name": "Pencil Skirt Gray", "category": "Clothing", "price": 49.99, "color": "gray", "material": "wool blend", "description": "Professional gray pencil skirt", "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400"},
    {"id": "33", "name": "Pleated Mini Skirt", "category": "Clothing", "price": 39.99, "color": "navy", "material": "polyester", "description": "Cute navy pleated mini skirt", "image_url": "https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400"},
    
    # CLOTHING - JACKETS & OUTERWEAR
    {"id": "34", "name": "Leather Jacket Black", "category": "Clothing", "price": 199.99, "color": "black", "material": "leather", "description": "Classic black leather motorcycle jacket", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "35", "name": "Denim Jacket Blue", "category": "Clothing", "price": 79.99, "color": "blue", "material": "denim", "description": "Vintage blue denim jacket", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "36", "name": "Bomber Jacket Green", "category": "Clothing", "price": 89.99, "color": "olive green", "material": "nylon", "description": "Military-inspired olive green bomber jacket", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "37", "name": "Blazer Navy Blue", "category": "Clothing", "price": 129.99, "color": "navy", "material": "wool", "description": "Formal navy blue blazer", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "38", "name": "Puffer Jacket Black", "category": "Clothing", "price": 149.99, "color": "black", "material": "nylon", "description": "Warm black puffer jacket for winter", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "39", "name": "Trench Coat Beige", "category": "Clothing", "price": 179.99, "color": "beige", "material": "cotton", "description": "Classic beige trench coat", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    {"id": "40", "name": "Cardigan Sweater Gray", "category": "Clothing", "price": 64.99, "color": "gray", "material": "wool", "description": "Cozy gray cardigan sweater", "image_url": "https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400"},
    
    # FOOTWEAR - SNEAKERS & ATHLETIC
    {"id": "41", "name": "Running Shoes Black", "category": "Footwear", "price": 89.99, "color": "black", "material": "mesh", "description": "High-performance running shoes with cushioning", "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400"},
    {"id": "42", "name": "White Sneakers", "category": "Footwear", "price": 79.99, "color": "white", "material": "canvas", "description": "Classic white canvas sneakers", "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400"},
    {"id": "43", "name": "Gray Running Shoes", "category": "Footwear", "price": 89.99, "color": "gray", "material": "mesh", "description": "Performance gray running shoes with foam sole", "image_url": "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=400"},
    {"id": "44", "name": "Blue Athletic Shoes", "category": "Footwear", "price": 79.99, "color": "blue", "material": "synthetic", "description": "Lightweight blue athletic shoes for sports", "image_url": "https://images.unsplash.com/photo-1551107696-a4b0c5a0d9a2?w=400"},
    {"id": "45", "name": "Red Casual Shoes", "category": "Footwear", "price": 69.99, "color": "red", "material": "canvas", "description": "Casual red canvas shoes for everyday", "image_url": "https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=400"},
    {"id": "46", "name": "High-Top Sneakers", "category": "Footwear", "price": 94.99, "color": "black", "material": "canvas", "description": "Classic black high-top sneakers", "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400"},
    {"id": "47", "name": "Basketball Shoes", "category": "Footwear", "price": 119.99, "color": "red", "material": "synthetic", "description": "Professional basketball shoes with ankle support", "image_url": "https://images.unsplash.com/photo-1551107696-a4b0c5a0d9a2?w=400"},
    {"id": "48", "name": "Tennis Shoes White", "category": "Footwear", "price": 84.99, "color": "white", "material": "leather", "description": "Classic white tennis shoes", "image_url": "https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400"},
    {"id": "49", "name": "Skateboard Shoes", "category": "Footwear", "price": 74.99, "color": "gray", "material": "suede", "description": "Durable gray skateboard shoes", "image_url": "https://images.unsplash.com/photo-1606107557195-0e29a4b5b4aa?w=400"},
    {"id": "50", "name": "Cross Training Shoes", "category": "Footwear", "price": 99.99, "color": "multicolor", "material": "mesh", "description": "Versatile cross training shoes", "image_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400"},
    
    # FOOTWEAR - BOOTS & FORMAL
    {"id": "51", "name": "Black Leather Boots", "category": "Footwear", "price": 129.99, "color": "black", "material": "leather", "description": "Durable black leather boots for any season", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "52", "name": "Brown Leather Shoes", "category": "Footwear", "price": 99.99, "color": "brown", "material": "leather", "description": "Classic brown leather dress shoes", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "53", "name": "White Leather Boots", "category": "Footwear", "price": 119.99, "color": "white", "material": "leather", "description": "Stylish white leather boots", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "54", "name": "Combat Boots Black", "category": "Footwear", "price": 139.99, "color": "black", "material": "leather", "description": "Military-style black combat boots", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "55", "name": "Chelsea Boots Brown", "category": "Footwear", "price": 149.99, "color": "brown", "material": "suede", "description": "Elegant brown suede Chelsea boots", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "56", "name": "Oxford Dress Shoes", "category": "Footwear", "price": 159.99, "color": "black", "material": "leather", "description": "Formal black Oxford dress shoes", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "57", "name": "Loafers Brown", "category": "Footwear", "price": 89.99, "color": "brown", "material": "leather", "description": "Comfortable brown leather loafers", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "58", "name": "Hiking Boots", "category": "Footwear", "price": 169.99, "color": "brown", "material": "leather", "description": "Waterproof hiking boots for outdoor adventures", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "59", "name": "Work Boots Steel Toe", "category": "Footwear", "price": 179.99, "color": "brown", "material": "leather", "description": "Heavy-duty work boots with steel toe", "image_url": "https://images.unsplash.com/photo-1544966503-7cc5ac882d5f?w=400"},
    {"id": "60", "name": "Rain Boots Yellow", "category": "Footwear", "price": 39.99, "color": "yellow", "color_options": ["yellow", "red", "blue", "green", "black"], "material": "rubber", "description": "Waterproof yellow rain boots", "image_url": "https://cdn.pixabay.com/photo/2016/11/29/05/45/boots-1867336_960_720.jpg"},
    
    # ACCESSORIES - WALLETS & BELTS
    {"id": "61", "name": "Leather Wallet Brown", "category": "Accessories", "price": 49.99, "color": "brown", "material": "leather", "description": "Premium brown leather bifold wallet", "image_url": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400"},
    {"id": "62", "name": "Black Leather Wallet", "category": "Accessories", "price": 49.99, "color": "black", "material": "leather", "description": "Classic black leather wallet with card slots", "image_url": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400"},
    {"id": "63", "name": "Minimalist Wallet", "category": "Accessories", "price": 34.99, "color": "black", "material": "carbon fiber", "description": "Slim minimalist carbon fiber wallet", "image_url": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400"},
    {"id": "64", "name": "Money Clip Silver", "category": "Accessories", "price": 29.99, "color": "silver", "material": "stainless steel", "description": "Sleek stainless steel money clip", "image_url": "https://images.unsplash.com/photo-1627123424574-724758594e93?w=400"},
    {"id": "65", "name": "Brown Leather Belt", "category": "Accessories", "price": 39.99, "color": "brown", "material": "leather", "description": "Durable brown leather belt with buckle", "image_url": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400"},
    {"id": "66", "name": "Black Leather Belt", "category": "Accessories", "price": 39.99, "color": "black", "material": "leather", "description": "Classic black leather belt", "image_url": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400"},
    {"id": "67", "name": "Canvas Belt Khaki", "category": "Accessories", "price": 24.99, "color": "khaki", "material": "canvas", "description": "Casual khaki canvas belt", "image_url": "https://images.unsplash.com/photo-1594736797933-d0401ba2fe65?w=400"},
    
    # ACCESSORIES - WATCHES
    {"id": "68", "name": "Stainless Steel Watch", "category": "Accessories", "price": 149.99, "color": "silver", "material": "stainless steel", "description": "Elegant stainless steel wristwatch", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    {"id": "69", "name": "Black Analog Watch", "category": "Accessories", "price": 99.99, "color": "black", "material": "leather", "description": "Classic black watch with leather strap", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    {"id": "70", "name": "Digital Sports Watch", "category": "Accessories", "price": 79.99, "color": "black", "material": "rubber", "description": "Waterproof digital sports watch", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    {"id": "71", "name": "Rose Gold Watch", "category": "Accessories", "price": 189.99, "color": "rose gold", "material": "stainless steel", "description": "Luxury rose gold watch", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    {"id": "72", "name": "Vintage Pocket Watch", "category": "Accessories", "price": 129.99, "color": "gold", "material": "brass", "description": "Classic vintage pocket watch", "image_url": "https://images.unsplash.com/photo-1524592094714-0f0654e20314?w=400"},
    
    # ACCESSORIES - HATS & CAPS
    {"id": "73", "name": "Blue Baseball Cap", "category": "Accessories", "price": 24.99, "color": "blue", "color_options": ["blue", "red", "black", "white", "gray"], "material": "cotton", "description": "Comfortable blue cotton baseball cap", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    {"id": "74", "name": "Black Beanie Hat", "category": "Accessories", "price": 19.99, "color": "black", "material": "wool", "description": "Warm black wool beanie", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    {"id": "75", "name": "Snapback Cap Red", "category": "Accessories", "price": 29.99, "color": "red", "color_options": ["red", "blue", "black", "white", "gray"], "material": "cotton", "description": "Trendy red snapback cap", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    {"id": "76", "name": "Fedora Hat Gray", "category": "Accessories", "price": 59.99, "color": "gray", "material": "felt", "description": "Classic gray felt fedora hat", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    {"id": "77", "name": "Bucket Hat Khaki", "category": "Accessories", "price": 22.99, "color": "khaki", "material": "cotton", "description": "Casual khaki bucket hat", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    {"id": "78", "name": "Trucker Hat Mesh", "category": "Accessories", "price": 19.99, "color": "black", "material": "mesh", "description": "Classic black mesh trucker hat", "image_url": "https://images.unsplash.com/photo-1521369909029-2afed882baee?w=400"},
    
    # ACCESSORIES - JEWELRY
    {"id": "79", "name": "Silver Chain Necklace", "category": "Accessories", "price": 29.99, "color": "silver", "color_options": ["silver", "gold", "rose gold"], "material": "silver", "description": "Elegant silver chain necklace", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    {"id": "80", "name": "Gold Bracelet", "category": "Accessories", "price": 59.99, "color": "gold", "color_options": ["gold", "silver", "rose gold"], "material": "gold plated", "description": "Stylish gold plated bracelet", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    {"id": "81", "name": "Diamond Stud Earrings", "category": "Accessories", "price": 199.99, "color": "silver", "material": "sterling silver", "description": "Elegant diamond stud earrings", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    {"id": "82", "name": "Leather Bracelet", "category": "Accessories", "price": 24.99, "color": "brown", "material": "leather", "description": "Casual brown leather bracelet", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    {"id": "83", "name": "Pearl Necklace", "category": "Accessories", "price": 149.99, "color": "white", "material": "pearl", "description": "Classic white pearl necklace", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    {"id": "84", "name": "Signet Ring Gold", "category": "Accessories", "price": 89.99, "color": "gold", "material": "gold plated", "description": "Classic gold signet ring", "image_url": "https://images.unsplash.com/photo-1515562141207-7a88fb7ce338?w=400"},
    
    # BAGS - BACKPACKS
    {"id": "85", "name": "Backpack Laptop", "category": "Bags", "price": 79.99, "color": "black", "color_options": ["black", "gray", "navy"], "material": "nylon", "description": "Professional black laptop backpack with USB port", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop"},
    {"id": "86", "name": "School Backpack Blue", "category": "Bags", "price": 49.99, "color": "blue", "color_options": ["blue", "red", "green", "black"], "material": "polyester", "description": "Durable blue school backpack", "image_url": "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?w=400&h=400&fit=crop"},
    {"id": "87", "name": "Gray Travel Backpack", "category": "Bags", "price": 99.99, "color": "gray", "material": "polyester", "description": "Large gray travel backpack with compartments", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop"},
    {"id": "88", "name": "Hiking Backpack Green", "category": "Bags", "price": 129.99, "color": "green", "material": "nylon", "description": "Outdoor hiking backpack with hydration system", "image_url": "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?w=400&h=400&fit=crop"},
    {"id": "89", "name": "Mini Backpack Pink", "category": "Bags", "price": 39.99, "color": "pink", "material": "leather", "description": "Cute pink mini backpack", "image_url": "https://images.unsplash.com/photo-1622560480605-d83c853bc5c3?w=400&h=400&fit=crop"},
    {"id": "90", "name": "Rolling Backpack", "category": "Bags", "price": 89.99, "color": "black", "material": "polyester", "description": "Convertible rolling backpack for travel", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400&h=400&fit=crop"},
    
    # BAGS - HANDBAGS & PURSES
    {"id": "91", "name": "Leather Shoulder Bag", "category": "Bags", "price": 89.99, "color": "brown", "material": "leather", "description": "Classic brown leather shoulder bag", "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400"},
    {"id": "92", "name": "Red Crossbody Bag", "category": "Bags", "price": 59.99, "color": "red", "color_options": ["red", "black", "brown", "navy"], "material": "polyester", "description": "Stylish red crossbody bag for everyday", "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400"},
    {"id": "93", "name": "Designer Handbag Black", "category": "Bags", "price": 199.99, "color": "black", "material": "leather", "description": "Luxury black designer handbag", "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400"},
    {"id": "94", "name": "Clutch Bag Gold", "category": "Bags", "price": 49.99, "color": "gold", "material": "synthetic", "description": "Elegant gold clutch bag for evening", "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400"},
    {"id": "95", "name": "Tote Bag Canvas", "category": "Bags", "price": 34.99, "color": "natural", "material": "canvas", "description": "Eco-friendly canvas tote bag", "image_url": "https://images.unsplash.com/photo-1584917865442-de89df76afd3?w=400"},
    
    # BAGS - BUSINESS & TRAVEL
    {"id": "96", "name": "Black Messenger Bag", "category": "Bags", "price": 69.99, "color": "black", "material": "canvas", "description": "Practical black canvas messenger bag", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
    {"id": "97", "name": "Black Leather Briefcase", "category": "Bags", "price": 129.99, "color": "black", "material": "leather", "description": "Professional black leather briefcase", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
    {"id": "98", "name": "Duffel Bag Navy", "category": "Bags", "price": 79.99, "color": "navy", "material": "nylon", "description": "Large navy duffel bag for gym or travel", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
    {"id": "99", "name": "Laptop Sleeve Gray", "category": "Bags", "price": 29.99, "color": "gray", "material": "neoprene", "description": "Protective gray laptop sleeve", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
    {"id": "100", "name": "Weekender Bag Brown", "category": "Bags", "price": 119.99, "color": "brown", "material": "leather", "description": "Stylish brown leather weekender bag", "image_url": "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=400"},
    
    # ELECTRONICS - AUDIO
    {"id": "101", "name": "Wireless Earbuds", "category": "Electronics", "price": 79.99, "color": "black", "material": "plastic", "description": "True wireless earbuds with noise cancellation", "image_url": "https://images.unsplash.com/photo-1590658268037-6bf12165a8df?w=400&h=400&fit=crop"},
    {"id": "102", "name": "Bluetooth Speaker", "category": "Electronics", "price": 59.99, "color": "gray", "material": "plastic", "description": "Portable gray bluetooth speaker with bass", "image_url": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=400&fit=crop"},
    {"id": "103", "name": "Over-Ear Headphones", "category": "Electronics", "price": 149.99, "color": "black", "material": "plastic", "description": "Premium over-ear headphones with noise cancelling", "image_url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=400&fit=crop"},
    {"id": "104", "name": "Gaming Headset", "category": "Electronics", "price": 89.99, "color": "red", "material": "plastic", "description": "RGB gaming headset with microphone", "image_url": "https://images.unsplash.com/photo-1583394838336-acd977736f90?w=400&h=400&fit=crop"},
    {"id": "105", "name": "Soundbar", "category": "Electronics", "price": 199.99, "color": "black", "material": "plastic", "description": "Home theater soundbar with subwoofer", "image_url": "https://images.unsplash.com/photo-1608043152269-423dbba4e7e1?w=400&h=400&fit=crop"},
    {"id": "106", "name": "Vinyl Record Player", "category": "Electronics", "price": 299.99, "color": "wood", "material": "wood", "description": "Vintage-style vinyl record player", "image_url": "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?w=400&h=400&fit=crop"},
    
    # ELECTRONICS - MOBILE & ACCESSORIES
    {"id": "107", "name": "Phone Stand", "category": "Electronics", "price": 19.99, "color": "black", "material": "aluminum", "description": "Adjustable aluminum phone stand", "image_url": "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=400"},
    {"id": "108", "name": "Wireless Charger", "category": "Electronics", "price": 34.99, "color": "white", "material": "plastic", "description": "Fast wireless charging pad", "image_url": "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=400"},
    {"id": "109", "name": "Phone Case Clear", "category": "Electronics", "price": 24.99, "color": "clear", "material": "silicone", "description": "Transparent protective phone case", "image_url": "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=400"},
    {"id": "110", "name": "Car Phone Mount", "category": "Electronics", "price": 29.99, "color": "black", "material": "plastic", "description": "Magnetic car phone mount", "image_url": "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=400"},
    {"id": "111", "name": "Selfie Stick", "category": "Electronics", "price": 19.99, "color": "black", "material": "aluminum", "description": "Extendable selfie stick with bluetooth", "image_url": "https://images.unsplash.com/photo-1512499617640-c74ae3a79d37?w=400"},
    
    # ELECTRONICS - CABLES & POWER
    {"id": "112", "name": "USB-C Cable", "category": "Electronics", "price": 14.99, "color": "black", "material": "plastic", "description": "High-speed USB-C charging cable", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
    {"id": "113", "name": "Power Bank 20000mAh", "category": "Electronics", "price": 39.99, "color": "black", "material": "plastic", "description": "Fast charging 20000mAh power bank", "image_url": "https://images.unsplash.com/photo-1609592806596-4d8b6b1b0b0b?w=400"},
    {"id": "114", "name": "Wall Charger Fast", "category": "Electronics", "price": 24.99, "color": "white", "material": "plastic", "description": "65W fast wall charger with multiple ports", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
    {"id": "115", "name": "Lightning Cable", "category": "Electronics", "price": 19.99, "color": "white", "material": "plastic", "description": "MFi certified lightning cable", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
    {"id": "116", "name": "Extension Cord", "category": "Electronics", "price": 29.99, "color": "black", "material": "plastic", "description": "6-outlet surge protector extension cord", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
    
    # ELECTRONICS - COMPUTER ACCESSORIES
    {"id": "117", "name": "Wireless Mouse", "category": "Electronics", "price": 34.99, "color": "black", "material": "plastic", "description": "Ergonomic wireless mouse", "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400"},
    {"id": "118", "name": "Mechanical Keyboard", "category": "Electronics", "price": 89.99, "color": "black", "material": "aluminum", "description": "RGB mechanical gaming keyboard", "image_url": "https://images.unsplash.com/photo-1541140532154-b024d705b90a?w=400"},
    {"id": "119", "name": "Webcam HD", "category": "Electronics", "price": 49.99, "color": "black", "material": "plastic", "description": "1080p HD webcam with microphone", "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400"},
    {"id": "120", "name": "USB Hub", "category": "Electronics", "price": 24.99, "color": "silver", "material": "aluminum", "description": "7-port USB 3.0 hub", "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400"},
    {"id": "121", "name": "Monitor Stand", "category": "Electronics", "price": 39.99, "color": "black", "material": "aluminum", "description": "Adjustable monitor stand with storage", "image_url": "https://images.unsplash.com/photo-1527864550417-7fd91fc51a46?w=400"},
    
    # ELECTRONICS - SMART DEVICES
    {"id": "122", "name": "Smart Watch", "category": "Electronics", "price": 199.99, "color": "black", "material": "aluminum", "description": "Fitness tracking smart watch", "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"},
    {"id": "123", "name": "Fitness Tracker", "category": "Electronics", "price": 79.99, "color": "blue", "material": "silicone", "description": "Waterproof fitness activity tracker", "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400"},
    {"id": "124", "name": "Smart Home Hub", "category": "Electronics", "price": 99.99, "color": "white", "material": "plastic", "description": "Voice-controlled smart home hub", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
    {"id": "125", "name": "Security Camera", "category": "Electronics", "price": 129.99, "color": "white", "material": "plastic", "description": "WiFi security camera with night vision", "image_url": "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=400"},
]


def build_index_with_products(use_real_data=False):
    """Build enhanced FAISS index with more products"""
    
    print("\n" + "="*70)
    print("Building Enhanced Product Search Index")
    print("="*70)
    
    # Setup paths
    data_dir = Path("data")
    index_dir = data_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "products"
    
    # Load dataset
    print(f"\n📦 Loading product database...")
    
    if use_real_data:
        # Try to load from CSV/JSON
        csv_path = data_dir / "products.csv"
        json_path = data_dir / "products.json"
        
        if csv_path.exists():
            import pandas as pd
            df = pd.read_csv(csv_path)
            products = df.to_dict('records')
            print(f"✓ Loaded {len(products)} products from CSV")
        elif json_path.exists():
            with open(json_path) as f:
                products = json.load(f)
            print(f"✓ Loaded {len(products)} products from JSON")
        else:
            print(f"⚠ No CSV/JSON found, using built-in database")
            products = PRODUCTS_DATABASE
    else:
        products = PRODUCTS_DATABASE
    
    print(f"✓ Total products: {len(products)}")
    
    # Initialize encoder
    print("\n🤖 Loading CLIP model...")
    encoder = CLIPEncoder(model_name="ViT-B/32")
    
    # Generate embeddings
    print(f"\n🔄 Generating embeddings for {len(products)} products...")
    embeddings = []
    
    for product in tqdm(products, desc="Encoding products"):
        # Create rich text description for better embeddings
        text = f"{product['name']} {product.get('category', '')} {product.get('color', '')} {product.get('description', '')}"
        embedding = encoder.encode_text(text)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Build FAISS index
    print("\n🔍 Building FAISS index...")
    faiss_index = FAISSIndex(embedding_dim=encoder.get_embedding_dim())
    faiss_index.build_index(
        embeddings=embeddings,
        metadata=products,
        index_type="HNSW",
        M=64,  # More connections for better accuracy
        ef_construction=400  # Higher quality search
    )
    
    # Save index
    print(f"\n💾 Saving index...")
    faiss_index.save(str(index_path))
    
    # Save product catalog
    catalog_path = index_dir / "catalog.json"
    with open(catalog_path, 'w') as f:
        json.dump(products, f, indent=2)
    print(f"✓ Saved product catalog")
    
    # Print statistics
    print("\n" + "="*70)
    print("✅ Index Building Complete!")
    print("="*70)
    print(f"\n📊 Statistics:")
    print(f"   Total products: {len(products)}")
    print(f"   Index size: {index_path.with_suffix('.index').stat().st_size / 1024 / 1024:.2f} MB")
    print(f"   Embedding dimension: {encoder.get_embedding_dim()}")
    print(f"\n📁 Categories: {', '.join(set(p['category'] for p in products))}")
    print(f"💰 Price range: ${min(p['price'] for p in products):.2f} - ${max(p['price'] for p in products):.2f}")
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build enhanced product search index")
    parser.add_argument('--real-data', action='store_true', help='Load from CSV/JSON')
    
    args = parser.parse_args()
    build_index_with_products(use_real_data=args.real_data)