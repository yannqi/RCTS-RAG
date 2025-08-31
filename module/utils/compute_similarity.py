
import numpy as np 
import pandas as pd
import faiss
from torch.nn.functional import cosine_similarity

def sort_captions_based_on_similarity(captions,raw_image,model,processor, device = "cuda", ascending = False):
    """
    Rank the qr captions based on their similarity with the image
    :param captions: The captions that will be ranked 
    :param raw_image: The PIL image object 
    :param model: The image-to-text similarity model (BLIP)
    :param processor: The image and text processor 
    :param device: Cpu or Gpu
    :param ascending: Bool variable for ranking the captions at ascending order or not 
    :returns results_df: Captions ranked 
    :returns cosine_scores: The cosine score of each caption with the image
    """
    #encode the captions
    text_input = processor(text = captions, return_tensors="pt", padding = True).to(device)
    text_embeds = model.text_encoder(**text_input)
    text_embeds = text_embeds[0]
    text_features = model.text_proj(text_embeds[:, 0, :])

    #encode the image 
    image_input = processor(images=raw_image, return_tensors="pt").to(device)
    vision_outputs = model.vision_model(**image_input)
    image_embeds = vision_outputs[0]
    image_feat = model.vision_proj(image_embeds[:, 0, :])
    
    #compute cos sim
    cosine_scores = cosine_similarity(text_features, image_feat).tolist()

    #sort captions based on the cosine scores
    captions = [x for _, x in sorted(zip(cosine_scores, captions), reverse = True)]
    cosine_scores.sort(reverse = True)
    return captions, cosine_scores

def get_context_examples_withRRF(sample_q_embed, sample_i_embed, q_faiss_db, i_faiss_db, n_shots, index_type):
    """
    Get the n context examples for n-shot in context learning
    according to the avg img and question similarities

    """
    sample_i_embed = sample_i_embed.reshape(1,-1).numpy()
    sample_q_embed = sample_q_embed.reshape(1,-1).numpy()
    if index_type == 'cosine':
        faiss.normalize_L2(sample_i_embed)
        faiss.normalize_L2(sample_q_embed)
    else:
        raise ValueError('index_type must be cosine now')
    #compute question sims 
    distance_image, index_image = i_faiss_db.search(sample_i_embed, n_shots*2)
    
    #compute image sims 
    distance_text, index_text = q_faiss_db.search(sample_q_embed, n_shots*2)
    
    # USE RRF to rank the results
    index_list = np.concatenate((index_image, index_text), axis=0)
    index_RRF, distance_RRF = reciprocal_rank_fusion(index_list, k=60)
    index_RRF = index_RRF[:n_shots]
    return index_RRF




# Reciprocal Rank Fusion algorithm
def reciprocal_rank_fusion(index_list, k=60):
    fused_scores = {}
   
        
    for index_rank in index_list:
        for rank, (index) in enumerate(index_rank):
            if index not in fused_scores:
                fused_scores[index] = 0
            previous_score = fused_scores[index]
            fused_scores[index] += 1 / (rank + k)
    sorted_fused_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    reranked_results = {index: score for index, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}
    index_RRF = [index for index, score in reranked_results.items()]
    distance_RRF = [score for index, score in reranked_results.items()]
    return index_RRF, distance_RRF


def get_context_examples_withText(sample_q_embed, sample_i_embed, q_faiss_db, i_faiss_db, n_shots, index_type):
    """
    Get the n context examples for n-shot in context learning
    according to the avg img and question similarities

    """
    sample_i_embed = sample_i_embed.reshape(1,-1).numpy()
    sample_q_embed = sample_q_embed.reshape(1,-1).numpy()
    if index_type == 'cosine':
        faiss.normalize_L2(sample_i_embed)
        faiss.normalize_L2(sample_q_embed)
    else:
        raise ValueError('index_type must be cosine now')
    #compute question sims 
    distance_image, index_image = i_faiss_db.search(sample_i_embed, n_shots)
    
    #compute image sims 
    distance_text, index_text = q_faiss_db.search(sample_q_embed, n_shots)
    
    # USE RRF to rank the results
    # index_list = np.concatenate((index_image, index_text), axis=0)
    # index_RRF, distance_RRF = reciprocal_rank_fusion(index_list, k=60)
    # index_RRF = index_RRF[:n_shots]

    return index_text[0]

def get_context_examples_withImg(sample_q_embed, sample_i_embed, q_faiss_db, i_faiss_db, n_shots, index_type):
    """
    Get the n context examples for n-shot in context learning
    according to the avg img and question similarities

    """
    sample_i_embed = sample_i_embed.reshape(1,-1).numpy()
    sample_q_embed = sample_q_embed.reshape(1,-1).numpy()
    if index_type == 'cosine':
        faiss.normalize_L2(sample_i_embed)
        faiss.normalize_L2(sample_q_embed)
    else:
        raise ValueError('index_type must be cosine now')
    #compute question sims 
    distance_image, index_image = i_faiss_db.search(sample_i_embed, n_shots)
    
    #compute image sims 
    distance_text, index_text = q_faiss_db.search(sample_q_embed, n_shots)
    
    # USE RRF to rank the results
    # index_list = np.concatenate((index_image, index_text), axis=0)
    # index_RRF, distance_RRF = reciprocal_rank_fusion(index_list, k=60)
    # index_RRF = index_RRF[:n_shots]

    return index_image[0]

