  function retrainClassifier(){
                    
                      $('#retrain').html('<i class="fa fa-refresh fa-spin fa-3x fa-fw" style="font-size:12px;"></i> Retraining Database');

                      $.ajax({
                                  type: "POST",
                                  url: "{{ url_for('retrain_classifier') }}",
                                  data : {}, 
                                  success: function(results) {
                                    console.log(results.finished);
                                    $('#numbers').html(results.finished+ "training")
                                    $('#retrain').html('<i class="fa fa-refresh fa-fw"></i> Retrain Database');

                                  },
                                  error: function(error) {
                                    console.log(error);
                                  }
                           });


                }
            function removeFace(id){
                    
                      //var text=$('#' + id + '_input').html();
                      var values = id.split('_');

                      var name = values[0];
                      var camera = values[1];
                      var person =document.getElementById(name + "_" + camera); //remove person from div
                      person.parentNode.removeChild(person);

                      $('#numbers').html(name+ " :  removed");
                      $.ajax({
                                  type: "POST",
                                  url: "{{ url_for('remove_face') }}",
                                  data : {'predicted_name': name, 'camera': camera}, 
                                  success: function(results) {
                                    console.log(results);
                                    $('#numbers').html('face removed: ' + results.face_removed);
                                        
                                  },
                                  error: function(error) {
                                    console.log(error);
                                  }
                           });
                }

            function addFace(id){
                    
                      var values = id.split('_');

                      var name = values[0];
                      var camera = values[1];
                      if(values[2]=='trust') {
                            var newName = "trust";
                      }
                      else{
                            var newName = document.getElementById(name + "_" + camera + "_input").value;

                      }

                      
                      var person =document.getElementById(name + "_" + camera); //remove person from div
                      person.parentNode.removeChild(person);

                      $('#numbers').html(name+ " : " + newName);
                      $.ajax({
                                  type: "POST",
                                  url: "{{ url_for('add_face') }}",
                                  data : {'person_id': name, 'new_name': newName, 'camera' : camera}, 
                                  success: function(results) {
                                    console.log(results);

                                    $('#numbers').html('face added ' + results.face_added);


                                        
                                  },
                                  error: function(error) {
                                    console.log(error);
                                  }
                           });

                }

                function createAlert() {
                            //getting selected option from dropdowns
                            var e = document.getElementById("cameras");
                            var cam = e.options[e.selectedIndex].value; 
                            var e1 = document.getElementById("event");
                            var eventd = e1.options[e1.selectedIndex].text; 
                            var e2 = document.getElementById("alarmstate");
                            var alarm = e2.options[e2.selectedIndex].text; 
                            var e3 = document.getElementById("people");
                            var pers = e3.options[e3.selectedIndex].text;  

                            var email = false;
                            var push = false;
                            var triggerA = false;
                            var notifyP = false;

                       
                            if(document.getElementById("email").checked==true) {
                                email = true;
                                alertstyle = "alert-success"
                            }
                            // if(document.getElementById("push").checked==true) {
                            //     push = true;
                            //     alertstyle = "alert-info"
                            // }
                            if(document.getElementById("trigger").checked==true) {
                                triggerA = true;
                                alertstyle = "alert-danger"
                            }
                            // if(document.getElementById("notify").checked==true) {
                            //     notifyP = true;
                            //     alertstyle = "alert-danger"
                            // }


                            //ajax post used to send alert data via json [ 'push_alert': push,'email_alert':email, 'trigger_alarm':triggerA, 'notify_police':notifyP]
                             $.ajax({
                                  type: "POST",
                                  url: "{{ url_for('create_alert') }}",
                                  data : {'camera': cam, 'eventdetail': eventd, 'alarmstate': alarm, 'person': pers, 'push_alert': push,'email_alert':email, 'trigger_alarm':triggerA,'notify_police':notifyP}, 
                                  success: function(results) {
                                    console.log(results);
                                    $('#numbers').html(results.alert_id);


                                        var alertdiv = document.createElement("div");
                                        alertdiv.setAttribute("class","alert alert-dismissable " + alertstyle);   
                                        var btn = document.createElement("BUTTON");        
                                        btn.setAttribute("type","button");    
                                        btn.setAttribute("class", "close"); 
                                        btn.setAttribute("data-dismiss", "alert");  
                                        btn.setAttribute("aria-hidden","true"); 
                                        btn.setAttribute("onclick","removeAlert(this.id)"); 
                                        btn.setAttribute("id",results.alert_id);  
                                        btn.innerHTML = "&times;";
                                        alertdiv.innerHTML = results.alert_message;
                                        alertdiv.appendChild(btn);
                                        document.getElementById("alert-list").appendChild(alertdiv);

                                  },
                                  error: function(error) {
                                    console.log(error);
                                  }
                            });

                            document.getElementById("numbers").innerHTML = "push: " + push+ " email: " + email+ " trigger: " + triggerA + "   " + push + "   "+email+ "   "+triggerA;
                }

                function removeAlert(id) {
                     $('#numbers').html("removed alert " + id);

                      $.ajax({
                              type: "POST",
                              url: "{{ url_for('remove_alert') }}",
                              data : {'alert_id': id}, 
                              success: function(results) {
                                console.log(results);
                                $('#numbers').html(results.alert_status);
                              },
                              error: function(error) {
                                console.log(error);
                              }
                        });

                 }

                function changeAlarmState() {

                            document.getElementById("numbers").innerHTML = "AlarmStateChanged";
                }
                function panic() {
                            document.getElementById("numbers").innerHTML = "Panic";
                }

                $(document).ready(function(){

                    var socket = io.connect('http://' + document.domain + ':' + location.port + '/test');

                    var numbers_received = [];
                    var people_received = [];

                     $("#changestate").click(function(){
                        socket.emit('alarm_state_change');
                        return false;
                     }); 

                    $("#panic").click(function(){
                        socket.emit('panic');
                        return false;
                     }); 

                    socket.on('my response', function(msg) {           //socket.on used to define event handeler
                        $('#log').append('<p>' + msg.data + '</p>');
                    });



                    $('form#emit').submit(function(event) {
                        
                    });

                    $('form#broadcast').submit(function(event) {
                        socket.emit('my broadcast event', {data: $('#broadcast_data').val()});
                        return false;
                    });

                     //receive details from server
                    socket.on('newnumber', function(msg) {
                        console.log("Received number" + msg.number);
                        //maintain a list of ten numbers
                        if (numbers_received.length >= 10){
                            numbers_received.shift()
                        }
                        numbers_received.push(msg.number);
                        numbers_string = '';
                        for (var i = 0; i < numbers_received.length; i++){
                            numbers_string = numbers_string + '<p>' + numbers_received[i].toString() + '</p>';
                        }
                        $('#numbers').html(numbers_string);
                    });

                     socket.on('people_detected', function(json) {   
                   
                        console.log("Received peopledata in Loop" + json);
                        var people = JSON.parse(json);
                        people_string = '';
                       
                        for (var i = 0; i < people.length; i++){

                            if(!document.getElementById(people[i].identity + "_" + people[i].camera)){


                                    var img_string = "/get_faceimg/"+ people[i].identity +"_" + people[i].camera + "#";

            /////////////////////////////////////////////////////////////////// Main divs
                                    var maindiv = document.createElement("div");
                                    maindiv.setAttribute("class", "col-md-6 ");
                                    maindiv.setAttribute("id", people[i].identity + "_"+ people[i].camera);
                                    var thumbdiv = document.createElement("div");
                                    thumbdiv.setAttribute("class", "thumbnail");
            /////////////////////////////////////////////////////////////////// img element
                                    var imgj = document.createElement("img");
                                    imgj.setAttribute("src", img_string + new Date().getTime());
                                    imgj.setAttribute("height", "50");
                                    imgj.setAttribute("width", "50");
                                    imgj.setAttribute("id", people[i].identity + "_" + people[i].camera + "_image");
                                    imgj.setAttribute("alt", "Random Name");
                                    imgj.setAttribute("class", "person"); //img-circle 
                                    //document.getElementById("progressbar").appendChild(imgj);
                                    thumbdiv.appendChild(imgj);
            /////////////////////////////////////////////////////////////////// name element
                                    var name = document.createElement("p");
                                    name.setAttribute("class", "text-center predictedName");
                                    name.setAttribute("id", people[i].identity + "_"+ people[i].camera+"_prediction");
                                    name.innerHTML = "<strong>"+people[i].prediction+"</strong>";
                                    thumbdiv.appendChild(name);
            /////////////////////////////////////////////////////////////////// button element
                                    var aligndiv = document.createElement("div");
                                    aligndiv.setAttribute("class","pull-right");   

                                    var btndiv = document.createElement("div");
                                    btndiv.setAttribute("class","btn-group");   
                                    var btn = document.createElement("BUTTON");        
                                    btn.setAttribute("type","button");    
                                    btn.setAttribute("class", "btn btn-default btn-xs dropdown-toggle");  
                                    btn.setAttribute("data-toggle","dropdown");  

                                    var spn = document.createElement("span");
                                    spn.setAttribute("class","caret");    
                                    btn.appendChild(spn);
                                   
                                    var list = document.createElement("ul");
                                    list.setAttribute("class","dropdown-menu text-centre"); 
                                    list.setAttribute("role","menu"); 
                                    list.setAttribute("id","faceActionList"); 


                                    var listitem = document.createElement("li");
                                    var inner = document.createElement("a");
                                    inner.setAttribute("id",people[i].identity + "_"+ people[i].camera+"_remove"); 
                                    inner.setAttribute("onclick","removeFace(this.id)"); 
                                    inner.innerHTML = "Remove";
                                    //addFace(id)
                                    listitem.appendChild(inner);
                                    list.appendChild(listitem);

                                    var listitem = document.createElement("li");
                                    var inner = document.createElement("a");
                                    inner.setAttribute("id",people[i].identity + "_"+ people[i].camera+"_trust"); 
                                    inner.setAttribute("onclick","addFace(this.id)"); 
                                    inner.innerHTML = "Trust";
                                    
                                    listitem.appendChild(inner);
                                    list.appendChild(listitem);

                                    var listitem = document.createElement("li");
                                    var inner = document.createElement("a");
                                    inner.setAttribute("data-toggle","modal"); 
                                    inner.setAttribute("id","addfacebtnID"); 
                                    inner.setAttribute("data-target","#"+people[i].identity + "_"+ people[i].camera+"_modal"); 
                                    inner.innerHTML = "Add New Face";
                                  
                                    listitem.appendChild(inner);
                                    list.appendChild(listitem);   

                                    btndiv.appendChild(btn);
                                    btndiv.appendChild(list);
                                    aligndiv.appendChild(btndiv);

                                    //document.getElementById("progressbar").appendChild(btndiv);   
                                    thumbdiv.appendChild(aligndiv);
             ///////////////////////////////////////////////////////////////////  progress bar element                     
                                    var d1 = document.createElement("div");
                                    d1.setAttribute("class","progress");
                                    var d2 = document.createElement("div");
                                    //var values = people[i].prediction.split('_');
                                    //var name = values[0];
                                    var conf = people[i].confidence; 
                                    console.log("New Person: " + people[i].prediction + ":"+people[i].confidence); 
                                    if(people[i].prediction != "unknown"){
                                        d2.setAttribute("class","progress-bar progress-bar-success");
                                        d2.setAttribute("role","progress-bar progress-bar-success");   
                                    }
                                    else{
                                           
                                        d2.setAttribute("class","progress-bar progress-bar-danger");
                                        d2.setAttribute("role","progress-bar progress-bar-danger");
                                        conf = 100 - people[i].confidence;
                                    }         
                                    
                                    d2.setAttribute("aria-valuenow","50");
                                    d2.setAttribute("aria-valuemin","0");
                                    d2.setAttribute("aria-valuemax","100");
                                    d2.setAttribute("style","width:" + conf +"%");
                                    d2.innerHTML = conf + "%";
                                    d1.appendChild(d2);

                                    var info = document.createElement("span");        
                                    info.setAttribute("id","detectioinfo"); 
                                    info.setAttribute("style","text-align:center; font-size:70%;");         
                                    info.innerHTML = "Camera " + people[i].camera + "  -  " + people[i].timeD;       
                                              
                           
             /////////////////////////////////////////////////////////////////////////////////////
                                 
                                    var modal = document.createElement("div");
                                    modal.setAttribute("class","modal fade col-md-3 text-center");
                                    modal.setAttribute("id",people[i].identity + "_" + people[i].camera + "_modal");
                                    modal.setAttribute("tabindex","-1");
                                    modal.setAttribute("role","dialog");
                                    modal.setAttribute("aria-labelledby","myModalLabel");
                                    modal.setAttribute("aria-hidden","true");
                                


                                    var dialog = document.createElement("div");
                                    dialog.setAttribute("class","modal-dialog");
                                    //modal.appendChild(dialog);

                                    var content = document.createElement("div");
                                    dialog.setAttribute("class","modal-content");
                                    //dialog.appendChild(content);

                                    var header = document.createElement("div");
                                    header.setAttribute("class","modal-header");
                                    //dialog.appendChild(content);

                                    var btn2 = document.createElement("BUTTON");        
                                    btn2.setAttribute("type","button");    
                                    btn2.setAttribute("class", "close");  
                                    btn2.setAttribute("data-dismiss","modal");  
                                    btn2.setAttribute("aria-hidden","true");  
                                    btn2.innerHTML = "&times;";

                                    header.appendChild(btn2);

                                    var title = document.createElement("h4");        
                                    title.setAttribute("id","myModalLabel");    
                                    title.setAttribute("class", "modal-title");  
                                    title.innerHTML = "Add face to Database";
                                  
                                    header.appendChild(title);                
                                    content.appendChild(header);

                                    var body = document.createElement("div");
                                    body.setAttribute("class","modal-body");

                                    var imgj1 = document.createElement("img");
                                    imgj1.setAttribute("src", img_string + new Date().getTime());
                                    imgj1.setAttribute("height", "100");
                                    imgj1.setAttribute("width", "100");
                                    imgj1.setAttribute("id", people[i].identity + "_" + people[i].camera+ "_imageModal");
                                    imgj1.setAttribute("class", "person"); //img-circle 
                                    
                                    body.appendChild(imgj1);

                                    var name = document.createElement("h4");
                                    name.setAttribute("class", "text-center");
                                    name.setAttribute("id", people[i].identity + "_" + people[i].camera+ "nameID");
                                    var values = people[i].prediction.split('_');
                                    var nameprediction = values[0];
                                    name.innerHTML = "<strong>" + nameprediction + " ?</strong>";

                                    body.appendChild(name);

                                    var form = document.createElement("div");
                                    form.setAttribute("class","form-group has-success");

                                    var input = document.createElement("input");
                                    input.setAttribute("class","form-control"); 
                                    input.setAttribute("placeholder","Enter Name"); 
                                    input.setAttribute("type","text");  
                                    input.setAttribute("id", people[i].identity + "_" + people[i].camera+ "_input");   

                                    form.appendChild(input);                
                                    body.appendChild(form);
                                 
                                    content.appendChild(body);

                                    var footer = document.createElement("div");
                                    footer.setAttribute("class","modal-footer");
                                    //dialog.appendChild(content);

                                    var btn3 = document.createElement("BUTTON");        
                                    btn3.setAttribute("type","button");    
                                    btn3.setAttribute("class", "btn btn-success pull-right");  
                                    btn3.setAttribute("data-dismiss","modal");  
                                    btn3.setAttribute("aria-hidden","true"); 
                                    btn3.setAttribute("id",people[i].identity + "_" + people[i].camera + "_add");  
                                    btn3.setAttribute("onclick","addFace(this.id)");  
                                    btn3.innerHTML = "Add Face";

                                    footer.appendChild(btn3);

                                    content.appendChild(footer);
                                    dialog.appendChild(content);
                                    modal.appendChild(dialog);




                                    thumbdiv.appendChild(d1);
                                    thumbdiv.appendChild(info);  
                                   
                                    maindiv.appendChild(thumbdiv);
                            

                                    document.getElementById("detected-faces").appendChild(maindiv);
                                    document.getElementById("detected-faces").appendChild(modal);
                            }
                            else{

                                var x = document.getElementById(people[i].identity+ "_" + people[i].camera).getElementsByClassName("progress-bar");
                                    
                                for (var j = 0; j < x.length; j++) {
                                        console.log("Updating detected face");
                                        var values = people[i].prediction.split('_');
                                        var name = values[0];
                                        var conf = people[i].confidence; 
                                        ///////////////////////
                                        if(name != "unknown"){
                                            x[j].setAttribute("class","progress-bar progress-bar-success");
                                            x[j].setAttribute("role","progress-bar progress-bar-success");
                                            var y = document.getElementById(people[i].identity+ "_" + people[i].camera).getElementsByClassName("predictedName");
                                            y[0].innerHTML = "<strong>"+people[i].prediction+"</strong>";
                                        
                                        }
                                        else{
                                               
                                            // x[j].setAttribute("class","progress-bar progress-bar-failure");
                                            // x[j].setAttribute("role","progress-bar progress-bar-failure");
                                            conf = 100 - people[i].confidence; 
                                        }       

                                        //////////////////////                          
                                        x[j].setAttribute("style","width:" + conf +"%");
                                        x[j].innerHTML = conf +"%";
                                    }
                                console.log("Updating image: " + people[i].identity);  
                                var img_string = "/get_faceimg/"+people[i].identity +'_'+ people[i].camera+ "#";
                                $('#' + people[i].identity + "_" + people[i].camera+ '_image').attr('src',  img_string  + new Date().getTime()); //jquery used to update image
                                $('#' + people[i].identity + "_" + people[i].camera+ "_imageModal").attr('src',  img_string  + new Date().getTime()); //update modal image 
                            }

                            //console.log(people_string);
              
                        }   

                        
                    });
                    
                      socket.on('alarm_status', function(json) {   

                               console.log("Alarm Status: " + json);
                               var alarm_status = JSON.parse(json);

                               if(alarm_status.triggered == true){

                                    $("#alarmStatus").html("Alarm Triggered");
                               }
                               else{
                                    $("#alarmStatus").html(alarm_status.state);

                               }         

                        });

                      socket.on('system_data', function(json) {   

                               console.log("System Data: " + json);
                               var system_data = JSON.parse(json);
                               var person;
                               var i = 0;
                               people_string = '';
                               for (; i < system_data.people.length;i++) {
                                    people_string = people_string + '<option>' + system_data.people[i]+ '</option>';
                               }
                                $('#people').html(people_string);

                               var i = 0;
                               camera_string = '';
                               for (; i < system_data.camNum;i++) {
                                    camera_string = camera_string + '<option value="' + i.toString() + '"> Camera ' + i + '</option>';
                               }
                               camera_string = camera_string + '<option value="All">All</option>';
                                $('#cameras').html(camera_string);

                        });


                });
