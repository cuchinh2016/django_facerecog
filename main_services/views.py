from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import History

def main(request):
    # return HttpResponse("Hello world!")
    return render(request, 'main_face_recog.html')


def history_list(request):
    histories = History.objects.all()
    return render(request, 'history_list.html', {'histories': histories})
