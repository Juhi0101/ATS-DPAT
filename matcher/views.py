import base64
import io
import json
import logging
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from PyPDF2 import PdfReader

from src.similarity import ats_match

logger = logging.getLogger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {'.txt', '.pdf'}


def extract_text_from_pdf(file_bytes):
    try:
        with io.BytesIO(file_bytes) as stream:
            reader = PdfReader(stream)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return '\n'.join(text_parts)
    except Exception as e:
        logger.exception('Error extracting text from PDF: %s', e)
        raise


def parse_uploaded_file(file_name, file_base64):
    if not file_name or not file_base64:
        return ''

    extension = Path(file_name).suffix.lower()
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError('Unsupported file type.')

    file_bytes = base64.b64decode(file_base64)
    if extension == '.txt':
        return file_bytes.decode('utf-8', errors='replace')
    if extension == '.pdf':
        return extract_text_from_pdf(file_bytes)

    raise ValueError('Unsupported file type.')


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
    resume_text = ''
    job_description = ''
    resume_file_name = ''
    resume_file_base64 = ''
    job_file_name = ''
    job_file_base64 = ''

    if request.content_type and request.content_type.startswith('application/json'):
        try:
            payload = json.loads(request.body.decode('utf-8'))
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON payload.'}, status=400)
        resume_text = payload.get('resume_text', '')
        job_description = payload.get('job_description', '')
        resume_file_name = payload.get('resume_file_name', '')
        resume_file_base64 = payload.get('resume_file_base64', '')
        job_file_name = payload.get('job_file_name', '')
        job_file_base64 = payload.get('job_file_base64', '')
    else:
        resume_text = request.POST.get('resume_text', '')
        job_description = request.POST.get('job_description', '')
        resume_file_name = request.POST.get('resume_file_name', '')
        resume_file_base64 = request.POST.get('resume_file_base64', '')
        job_file_name = request.POST.get('job_file_name', '')
        job_file_base64 = request.POST.get('job_file_base64', '')

    try:
        if resume_file_name and resume_file_base64:
            resume_text = parse_uploaded_file(resume_file_name, resume_file_base64)
        if job_file_name and job_file_base64:
            job_description = parse_uploaded_file(job_file_name, job_file_base64)
    except ValueError as exc:
        return JsonResponse({'error': str(exc)}, status=400)
    except Exception:
        return JsonResponse({'error': 'Unable to parse uploaded file. Please upload a valid TXT or PDF document.'}, status=400)

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
