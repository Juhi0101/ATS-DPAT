import json
import logging

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from src.similarity import ats_match

logger = logging.getLogger(__name__)


def home(request):
    return render(request, 'matcher/index.html')


def ats_test(request):
    sample_resume = "Experienced Python developer with knowledge of machine learning and data analysis."
    sample_job = "Looking for a Python developer skilled in data analysis, machine learning, business intelligence, and deep learning."

    result = ats_match(sample_resume, sample_job)
    return JsonResponse(result)


@csrf_exempt
@require_POST
def analyze_resume(request):
    if request.content_type == 'application/json':
        try:
            payload = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload.'}, status=400)
        resume_text = payload.get('resume_text', '')
        job_description = payload.get('job_description', '')
    else:
        resume_text = request.POST.get('resume_text', '')
        job_description = request.POST.get('job_description', '')

    if not resume_text or not job_description:
        return JsonResponse(
            {'error': 'resume_text and job_description are required.'},
            status=400,
        )

    try:
        result = ats_match(resume_text, job_description)
        matched_skills = result['matched_skills']
        missing_skills = result['missing_skills']
        total_skills = len(matched_skills) + len(missing_skills)
    except Exception as e:
        logger.exception('Error while analyzing resume: %s', e)
        return JsonResponse(
            {'error': 'An unexpected error occurred while analyzing your resume. Please try again.'},
            status=500,
        )

    if total_skills == 0:
        explanation = 'No skills were detected in the job description or resume, so a skills match could not be calculated.'
    else:
        missing_list = ', '.join(missing_skills[:5])
        if not missing_skills:
            explanation = f'Your resume matches all {total_skills} required skills.'
        else:
            explanation = (
                f'Your resume matches {len(matched_skills)} out of {total_skills} required skills. '
                f'Missing key skills include {missing_list}.'
            )

    response_data = {
        'similarity_score': result['similarity_score'],
        'final_score': result['final_score'],
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'matched_keywords': result['matched_keywords'],
        'missing_keywords': result['missing_keywords'],
        'feature_importance': result.get('feature_importance', []),
        'explanation': explanation,
    }
    return JsonResponse(response_data)
