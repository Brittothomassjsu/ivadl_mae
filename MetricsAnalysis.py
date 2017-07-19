import json
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import sys

class annotationmetrics():
    """
    Metric Analysis Engine evaluates model's annotation w.r.t ground truth

      evaluation dimensions :
      1. Bounding box precision : IoU

      2. Classifier accuracy counts : true positive ,true negative , false positive , false negative (tp,tn,fp,fn)
                                      Precision (K-sampled ,11 point sampled)
                                      Recall (K-sampled ,11 point sampled)
                                      Average precision
                                      mean_average_precision for all classifiers in dataset

    """
    def __init__(self,gtobj=[],pobj=[],for_video=False):
        self.p_obj = pobj
        self.gt_obj = gtobj
        self.p_obj_eval = []
        self.gt_obj_eval = []
        self.eval_result = []
        self.for_video = for_video
        self.gt_obj_class = []
        self.total_bb =0

    def jsonload(self, gtfile,pfile):
        """
        :param pfile: PATH to json file  with predicted objects

        :param gtfile: PATH to json file with ground truth objects

        :return: true - if loaded successfully, false - if not loaded

        """
        if path.exists(pfile) and path.exists(gtfile):
            with open(pfile, 'r') as pf:
                self.p_obj = json.loads(pf.read())
            with open(gtfile, 'r') as gtf:
                self.gt_obj = json.loads(gtf.read())
            print("\n" + "Loaded model & user annotated objects from .json file")
            return True
        else:
            return False

    def jsonParse(self,arr_in,arr_out):
        """
        Re-structures json object list to list of frames with objects for easy metrics analysis

        :param arr_in: Input json array of objects to parse
        :param arr_out: Ouput array of frames with its objects list
        :return: void

        """
        frm_id = 0
        kframe_id = 0
        tpf = round(1.0 / 30.0,2)
        arr_in = sorted(arr_in, key=lambda k: k['keyframes'][kframe_id]['frame'])
        li_frameid = []
        #print(arr_in)

        for i in range(len(arr_in)):
            frame_id = int(round(arr_in[i]['keyframes'][kframe_id]['frame'], 2) / tpf)

            if frame_id not in li_frameid:
                arr_out.append([])    # Added frame 0
                frm_id +=1
                li_frameid.append(frame_id)

            if i == 0:
                frm_id = 0

            obj = {}
            obj['color'] = arr_in[i]['color']
            obj['type'] = arr_in[i]['type']
            obj['x'] = arr_in[i]['keyframes'][kframe_id]['x']
            obj['y'] = arr_in[i]['keyframes'][kframe_id]['y']
            obj['w'] = arr_in[i]['keyframes'][kframe_id]['w']
            obj['h'] = arr_in[i]['keyframes'][kframe_id]['h']
            obj['frame'] = round(arr_in[i]['keyframes'][kframe_id]['frame'], 2)
            obj['continueInterpolation'] = arr_in[i]['keyframes'][kframe_id]['continueInterpolation']

            try:
                obj['prob'] = arr_in[i]['keyframes'][kframe_id]['prob']
                obj['bbID'] = arr_in[i]['keyframes'][kframe_id]['bbID']
            except KeyError:
                obj['prob'] = -1
                obj['bbID'] = -1

            obj['detected'] = -1
            obj['BBA'] = -1
            obj['frameID'] = frame_id
            arr_out[frm_id].append(obj)
        print(arr_out[frm_id])

    def mapFrames(self,gt_obj_eval,p_obj_eval):
        """
        Compares frame timestamps of GT and Predicted frames and assigns frameID

        :param gt_obj_eval: GT object parsed list
        :param p_obj_eval: predicted object parsed list
        :return:

        """
        gtframes = len(gt_obj_eval)
        pframes = len(p_obj_eval)
        obj_id = 0

        for gf in range(gtframes):
            for pf in range(pframes):
                if gt_obj_eval[gf][obj_id]['frame'] == p_obj_eval[gf][obj_id]['frame'] and p_obj_eval[pf][obj_id]['frameID']  == -1:
                    for obj in range(len(gt_obj_eval[gf])):
                        gt_obj_eval[gf][obj]['frameID'] = gf
                    for obj in range(len(p_obj_eval[pf])):
                        p_obj_eval[pf][obj]['frameID']  = gf
                    break
        p_obj_eval = sorted(p_obj_eval, key=lambda k: k[obj_id]['frameID'])


    def mapObjects(self,gt_obj_eval,p_obj_eval):
        """
        Compares GT Objects and predicted objects and matches then to assign objectID with same frameID

        For each GTO, IoU with all predicted objects are found, and the one with max value is matched

        Both the objects are assigned with same ID.

        :param gt_obj_eval: GT object parsed list
        :param p_obj_eval: predicted object parsed list
        :return:
        """
        gtframes = len(gt_obj_eval)
        pframes = len(p_obj_eval)
        obj_id =0
        print(pframes)
        print(gtframes)
        print("Map Objects between same frame")

        for frameid in range(gtframes):

            if frameid < pframes or True:

                if gt_obj_eval[frameid][obj_id]['frameID'] == p_obj_eval[frameid][obj_id]['frameID']:

                    #print("Map Objects for frame: {0}".format(frameid)+"\n")
                    for g in range(len(gt_obj_eval[frameid])):
                        rlist = []
                        self.total_bb +=1
                        for p in range(len(p_obj_eval[frameid])):
                            rt = self.iou(gt_obj_eval,p_obj_eval,frameid,g,p)  #calculate IoU
                            rlist.append(rt)

                        #print("GT Object {0} IoU:{1}".format(g,rlist))
                        ratio = max(rlist)
                        idx = rlist.index(ratio)
                        #print("Max found at {0}:{1}".format(idx,ratio))

                        p_obj_eval[frameid][idx]['bbID']= self.total_bb
                        gt_obj_eval[frameid][g]['bbID'] = self.total_bb

                        if ratio >= 0.5:
                            p_obj_eval[frameid][idx]['detected'] = 1
                            p_obj_eval[frameid][idx]['BBA'] = 2   # 2 - good

                        elif 0.5 > ratio > 0:
                            p_obj_eval[frameid][idx]['detected'] = 1
                            p_obj_eval[frameid][idx]['BBA'] = 1    # 1- bad

                        elif ratio == 0:
                            p_obj_eval[frameid][idx]['detected'] = 0
                            p_obj_eval[frameid][idx]['BBA'] = 0  # 0- Missed
                    #p_obj_eval[frameid] = sorted(p_obj_eval[frameid], key=lambda k: k['prob'], reverse=True)

    def detectedGTObjects(self,gt_obj_eval,p_obj_eval):
        """
        Updates detected flag in GT to for mapped bounding boxes

        :param self: Object instance
        :return:
        """
        gtframes = len(gt_obj_eval)
        pframes = len(p_obj_eval)
        obj_id = 0

        for frameid in range(gtframes):

            if frameid < pframes or True:

                if gt_obj_eval[frameid][obj_id]['frameID'] == p_obj_eval[frameid][obj_id]['frameID']:

                    for i in range(len(gt_obj_eval[frameid])):
                        for j in range(len(p_obj_eval[frameid])):
                            if gt_obj_eval[frameid][i]['bbID'] == p_obj_eval[frameid][j]['bbID']:
                                gt_obj_eval[frameid][i]['detected']= 1
                                break


    def iou(self, gt_obj_eval,p_obj_eval,frameid, gtbb, pbb):
        """

        :param self:
        :param frameid: mapped frame id
        :param gtbb: GT object  bounding box index in frameid
        :param pbb:  predicted object bounding box index in frameid
        :return:
        """
        ratio = 0
        x1 = p_obj_eval[frameid][pbb]['x']
        y1 = p_obj_eval[frameid][pbb]['y']
        w1 = p_obj_eval[frameid][pbb]['w']
        h1 = p_obj_eval[frameid][pbb]['h']

        x2 = gt_obj_eval[frameid][gtbb]['x']
        y2 = gt_obj_eval[frameid][gtbb]['y']
        w2 = gt_obj_eval[frameid][gtbb]['w']
        h2 = gt_obj_eval[frameid][gtbb]['h']

        endx = max(x1 + w1, x2 + w2)
        startx = min(x1, x2)
        w = w1 + w2 - (endx - startx)

        endy = max(y1 + h1, y2 + h2)
        starty = min(y1, y2)
        h = h1 + h2 - (endy - starty)

        if w <= 0 or h <= 0:
            area = 0
        else:
            area = w * h
            area1 = w1 * h1
            area2 = w2 * h2
            ratio = area / (area1 + area2 - area)
        return ratio

    def calculateAccuracyCount(self,gt_obj_eval,p_obj_eval):
        """
        Computes tp,fp for every class in a Frame/video

        :param self:
        :param arr_in: input bounding box array per frame/video
        :return:
        """
        self.findObjectClasses(gt_obj_eval)
        gtframes = len(gt_obj_eval)
        pframes = len(p_obj_eval)
        obj_id = 0

        for frameid in range(gtframes): #for each frame

            if frameid < pframes or True:

                acc_cnt = {'frameID': frameid, 'mAP': -1, 'classes': []}
                mean_average_precision=0

                for id,objclass in enumerate(self.gt_obj_class): #for each class
                    class_accuracy = {'class': objclass.lower(), 'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0, 'precision': [], 'recall': [],'AP': -1}
                    fn = 0
                    avg = 0
                    delta_recall = []
                    plist = []
                    acc_cnt['classes'].append(class_accuracy)

                    for i in range(len(p_obj_eval[frameid])):
                        if p_obj_eval[frameid][i]['type'] == objclass:
                            plist.append(p_obj_eval[frameid][i])
                    plist = sorted(plist, key=lambda k: k['prob'], reverse=True)

                    for i in range(len(gt_obj_eval[frameid])):
                        if gt_obj_eval[frameid][i]['detected'] == -1 and gt_obj_eval[frameid][i]['type'] == objclass:
                            fn += 1

                    k = list(range(1,(len(plist)+1)))

                    for cutoff in k:
                        tp = 0
                        fp = 0
                        f = list(range(cutoff))
                        #print(f)
                        for obj in f:
                            for x,y in enumerate(gt_obj_eval[frameid]):
                                if plist[obj]['bbID'] == y['bbID']:
                                    gt = x
                                    break
                            if plist[obj]['type'].lower() == objclass.lower() and (plist[obj]['type'].lower() == gt_obj_eval[frameid][gt]['type'].lower()):
                                tp += 1
                            if (plist[obj]['type'].lower() == objclass.lower()) and ((plist[obj]['type'].lower() != gt_obj_eval[frameid][gt]['type'].lower())):
                                fp += 1

                        acc_cnt['classes'][id]['tp'] = tp
                        acc_cnt['classes'][id]['fp'] = fp
                        acc_cnt['classes'][id]['fn'] = fn

                        if tp != 0 or fp!=0:
                            acc_cnt['classes'][id]['precision'].append((tp / (fp+tp)))
                        else:
                            acc_cnt['classes'][id]['precision'].append(0)

                        if tp != 0 or fn!=0:
                            acc_cnt['classes'][id]['recall'].append(( tp / (fn + tp)))
                        else:
                            acc_cnt['classes'][id]['recall'].append(0)

                    for r in range(len(acc_cnt['classes'][id]['recall'])):
                        if r > 0:
                            delta_recall.append(acc_cnt['classes'][id]['recall'][r]-acc_cnt['classes'][id]['recall'][r-1])
                        elif r==0:
                            delta_recall.append(acc_cnt['classes'][id]['recall'][r])

                    for a in range(len(acc_cnt['classes'][id]['recall'])):
                        avg += acc_cnt['classes'][id]['precision'][a] * delta_recall[a]


                    mean_average_precision += avg
                    acc_cnt['classes'][id]['AP']=avg

                    #print("Precision Recall for class {0}".format(objclass) + "\n" + str(acc_cnt['classes'][id]['precision']) + "\n" + str(acc_cnt['classes'][id]['recall']))
                    #print(k)
                    #print(plist)
                    #print(delta_recall)
                    #print("AP for class {0}".format(objclass)+ str(avg)+ "\n")
                    #self.graph(acc_cnt['classes'][id]['recall'],acc_cnt['classes'][id]['precision'][:],objclass)

                mean_average_precision = mean_average_precision/len(self.gt_obj_class)
                acc_cnt['mAP'] = mean_average_precision
                self.eval_result.append(acc_cnt)
                #print("mAP for frame:{0}".format(frameid) + "\n" + str(mean_average_precision))

            # for i in range(len(self.eval_result)):
            #     print("Accuracy count for frame:{0}".format(i) + "\n" + str(self.eval_result[i]))

    def findObjectClasses(self,gt_obj_eval):
        """
        Finds the list of objects detected in ground truth
        :param gt_obj_eval: GT parsed list
        :return:
        """
        self.gt_obj_class=[]
        gtframes = len(gt_obj_eval)

        for frameid in range(gtframes):
            for i in range(len(gt_obj_eval[frameid])):
                if gt_obj_eval[frameid][i]['type'].lower() not in self.gt_obj_class:
                    self.gt_obj_class.append(gt_obj_eval[frameid][i]['type'].lower())
        print("\n" + "GT classes labels" + "\n" + str(self.gt_obj_class))

    def display(self,text,p_obj_eval=[],pframes=1):
        # Show Objects per frame
        print("\n"+text+" list per frame ")
        #pframes = len(p_obj_eval)

        if pframes>0:
            for i in range(pframes):
                if isinstance(p_obj_eval[i],list):
                    print("Frame {0}:{1}".format(i, len(p_obj_eval[i])))
                    for obj in range(len(p_obj_eval[i])):
                        print(text+" {0}: {1} ".format(obj, p_obj_eval[i][obj]))
                elif isinstance(p_obj_eval[i],dict):
                    print(text+" {0}: {1} ".format(i, p_obj_eval[i]))
        else:
            print("Empty List")

    def graph(self,r,p,cl):
        """
        Plots PR graph for each class - Todo - change the UI
        :param r: recall list
        :param p:  precision list
        :param cl: class label
        :return:
        """
        plt.clf()
        plt.plot(r, p, lw=2.0, color='navy',label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(cl)
        plt.legend(loc="lower left")
        plt.show()

    def run(self):
        """
        Runs the metrics and updates the evaluation result
        :return: self.eval_result
        """
        self.jsonParse(self.p_obj,self.p_obj_eval)
        self.jsonParse(self.gt_obj, self.gt_obj_eval)

        diff = len(self.gt_obj_eval)-len(self.p_obj_eval)
        if diff != 0:
            if diff > 0:
                dummy_frame = [self.p_obj_eval[0][0]]
                for i in range(diff):
                    self.p_obj_eval.append(dummy_frame)
            else:
                dummy_frame = [self.gt_obj_eval[0][0]]
                for i in range(diff):
                    self.gt_obj_eval.append(dummy_frame)

        self.mapFrames(self.gt_obj_eval,self.p_obj_eval)
        self.mapObjects(self.gt_obj_eval,self.p_obj_eval)
        self.detectedGTObjects(self.gt_obj_eval,self.p_obj_eval)

        if self.for_video:
            gt_video = [[]]
            for frame in range(len(self.gt_obj_eval)):
                for obj in range(len(self.gt_obj_eval[frame])):
                    gt_video[0].append(self.gt_obj_eval[frame][obj])
            self.gt_obj_eval = gt_video

            p_video = [[]]
            for frame in range(len(self.p_obj_eval)):
                for obj in range(len(self.p_obj_eval[frame])):
                    p_video[0].append(self.p_obj_eval[frame][obj])
            self.p_obj_eval = p_video

        self.calculateAccuracyCount(self.gt_obj_eval, self.p_obj_eval)

        #Dump the logs to log.txt
        self.display("Predicted Object", self.p_obj_eval,len(self.p_obj_eval))
        # self.display("GT Object", self.gt_obj_eval,len(self.gt_obj_eval))
        # self.display("Evaluation Result", self.eval_result,len(self.eval_result))

        #plot the graph for one frame
        for id in range(len(self.eval_result[0]['classes'])):
            self.graph(self.eval_result[0]['classes'][id]['recall'], self.eval_result[0]['classes'][id]['precision'][:], self.eval_result[0]['classes'][id]['class'])

        return self.eval_result

def main():
    gt_file = "video_WALSH_2s_GT.json"
    yolo_file = "video_WALSH_2s_YOLO.json"
    frcnn_file = "video_WALSH_2s_RCNN.json"
    call_api = True
    sys.stdout = open("log.txt", 'w')
    if not call_api:

        # if path.exists(frcnn_file) and path.exists(gt_file):
        #     metrics_rcnn = annotationmetrics()
        #     metrics_rcnn.jsonload(gt_file, frcnn_file)
        #     metrics_rcnn.run()
        # else:
        #     print("File Error : Path doesn't exists")

        if path.exists(yolo_file) and path.exists(gt_file):
            metrics_yolo = annotationmetrics()
            metrics_yolo.jsonload(gt_file, yolo_file)
            metrics_yolo.run()
        else:
            print("File Error : Path doesn't exists")

    else: #Test run
        print("Testing Metrics analysis with FastRCNN model for video")
        if path.exists(yolo_file) and path.exists(gt_file):
            with open(yolo_file, 'r') as pf:
                 ep_obj = json.loads(pf.read())
            with open(gt_file, 'r') as gtf:
                 egt_obj = json.loads(gtf.read())
            rfile = evaluate_metrics(egt_obj,ep_obj,True,0,1,[],[])
            if path.exists(rfile):
                print("\n"+"Evalutation result written to file eval_result.json")
        else:
            print("File Error : Path doesn't exists")
        sys.stdout.close()

def evaluate_metrics(gt_object=[],pred_object=[],forVideo = True,video_info=[],frameid=0,AP_filter=[],mAP_filter=[]):
    """
    :param pred_object:  List of predicted objects for all frames in a video
    :param gt_object:    List of ground truth objects for all frames in a video
    :param forVideo:     flag to evaluate video if true, else frame annotation
    :param video_info:   To do
    :param frameid:      frame ID to annotate
    :param AP_filter:    List of classes filtered to display AP - To Do
    :param mAP_filter:   List of classes filtered to display mAP - To do
    :return: evaluation results json objects

    """
    eval_file = "eval_result.json"

    if pred_object and gt_object:

        if forVideo:
            print("Evaluating Video file")
        else:
            print("Evaluating frames")

        metrics = annotationmetrics(gt_object,pred_object,forVideo)
        evalutation_result = metrics.run()

        with open(eval_file,'w') as resultfile:
            json.dump(evalutation_result,resultfile,separators=(',', ': '))

    return eval_file

if __name__ ==  "__main__":
    main()
